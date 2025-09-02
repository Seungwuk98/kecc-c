#include "kecc/ir/IRAnalyses.h"
#include "kecc/ir/IRBuilder.h"
#include "kecc/ir/IRInstructions.h"
#include "kecc/ir/IRTransforms.h"
#include "kecc/ir/IRTypes.h"
#include "kecc/ir/Instruction.h"
#include "kecc/ir/TypeAttributeSupport.h"
#include <map>

namespace kecc::ir {

namespace {
struct ConvertResult {

  static std::optional<ConvertResult> noChange() { return std::nullopt; }
  static ConvertResult create(FunctionT type,
                              std::map<size_t, size_t> convertMap) {
    return ConvertResult{type, std::move(convertMap)};
  }

  FunctionT changed;
  std::map<size_t, size_t> convertMap;
};

std::optional<ConvertResult> convertType(FunctionT type,
                                         const StructSizeMap &sizeMap) {
  assert(type && "Cannot convert a null type in OutlineMultipleResults");

  auto retTypes = type.getReturnTypes();
  assert(
      llvm::all_of(retTypes,
                   [&](Type t) {
                     auto [size, align] = t.getSizeAndAlign(sizeMap);
                     return size <= 8;
                   }) &&
      "All return types must be 8 bytes or smaller in OutlineMultipleResults");

  if (retTypes.size() <= 2)
    return ConvertResult::noChange();

  std::map<size_t, size_t> convertMap;
  llvm::SmallVector<Type> newRetTypes;
  llvm::SmallVector<Type> newArgTypes;

  for (auto idx = 0u; idx < 2u; ++idx)
    newRetTypes.emplace_back(retTypes[idx]);

  for (auto idx = 2; idx < retTypes.size(); ++idx) {
    auto retType = retTypes[idx];
    auto pointerType = PointerT::get(retType.getContext(), retType);
    newArgTypes.emplace_back(pointerType);
    convertMap[idx] = newArgTypes.size() - 1;
  }

  auto newType = FunctionT::get(type.getContext(), newRetTypes, newArgTypes);

  return ConvertResult::create(newType, std::move(convertMap));
}

static void convertCall(inst::Call call, const StructSizeMap &sizeMap) {
  auto func = call.getFunction();
  auto funcT =
      func.getType().cast<PointerT>().getPointeeType().cast<FunctionT>();

  auto converted = convertType(funcT, sizeMap);
  if (!converted)
    return;

  IRBuilder builder(call->getContext());
  builder.setInsertionPoint(
      call.getParentBlock()->getParentFunction()->getAllocationBlock());

  llvm::SmallVector<Value> newArgs;
  llvm::DenseMap<size_t, inst::LocalVariable> newLocalVars;

  for (auto [retIdx, phi] : converted->convertMap) {
    auto retType = funcT.getReturnTypes()[retIdx];
    auto pointerType = PointerT::get(call->getContext(), retType);
    auto newLocalVar = builder.create<inst::LocalVariable>({}, pointerType);
    newArgs.emplace_back(newLocalVar);
    newLocalVars[retIdx] = newLocalVar;
  }

  newArgs.append(call.getArguments().begin(), call.getArguments().end());

  builder.setInsertionPointBeforeInst(call.getStorage());

  auto castedFunc = builder.create<inst::TypeCast>(
      call->getRange(), func,
      PointerT::get(call.getContext(), converted->changed));
  auto newCall =
      builder.create<inst::Call>(call->getRange(), castedFunc, newArgs);

  for (auto idx = 0u, newi = 0u; idx < call->getResultSize(); ++idx) {
    if (auto it = converted->convertMap.find(idx);
        it == converted->convertMap.end()) {
      call.getResult(idx).replaceWith(newCall.getResult(newi++));
    }
  }

  for (auto [retIdx, _] : converted->convertMap) {
    auto load =
        builder.create<inst::Load>(call->getRange(), newLocalVars[retIdx]);
    call.getResult(retIdx).replaceWith(load);
  }

  call.getParentBlock()->remove(call.getStorage());
}

static void convertRet(inst::Return ret,
                       const llvm::DenseMap<size_t, Phi> &retMemory) {
  IRBuilder builder(ret->getContext());
  builder.setInsertionPointBeforeInst(ret.getStorage());

  llvm::SmallVector<Value> newRetValues;
  for (size_t i = 0u; i < ret.getValueSize(); ++i) {
    if (auto it = retMemory.find(i); it != retMemory.end()) {
      auto phi = it->second;
      auto value = ret.getValue(i);
      assert(phi.getType().cast<PointerT>().getPointeeType() ==
                 value.getType() &&
             "Return value type must match the phi type in "
             "OutlineMultipleResults");
      auto store = builder.create<inst::Store>(ret->getRange(), value, phi);
    } else {
      newRetValues.emplace_back(ret.getValue(i));
    }
  }

  auto newRet = builder.create<inst::Return>(ret->getRange(), newRetValues);
  ret->getParentBlock()->remove(ret.getStorage());
}

static void convertAllValues(Module *module, const StructSizeMap &sizeMap) {
  auto convertTFunc = [&](Type type) -> ReplaceResult<Type> {
    if (auto funcT = type.dyn_cast<FunctionT>()) {
      auto converted = convertType(funcT, sizeMap);
      if (converted)
        return {converted->changed, utils::LogicalResult::success()};
    }
    return {type, utils::LogicalResult::success()};
  };

  auto convertValueFn = [&](Value value) {
    auto type = value.getType();
    auto replaced = type.replace(convertTFunc);
    assert(replaced &&
           "Replaced type must not be null in OutlineMultipleResults");
    if (type != replaced)
      value.getImpl()->setType(replaced);
  };

  auto convertInstruction = [&](InstructionStorage *storage) {
    auto results = storage->getResults();
    for (auto result : results)
      convertValueFn(result);

    for (auto [idx, attr] : llvm::enumerate(storage->getAttributes())) {
      auto converted = attr.replace(convertTFunc);
      assert(converted &&
             "Replaced attribute must not be null in OutlineMultipleResults");
      if (converted != attr)
        storage->setAttribute(idx, converted);
    }
  };

  for (auto *inst : *module->getIR()->getConstantBlock())
    convertInstruction(inst);

  for (auto *func : *module->getIR()) {
    for (InstructionStorage *inst : *func->getAllocationBlock())
      convertInstruction(inst);

    func->walk([&](InstructionStorage *inst) -> WalkResult {
      convertInstruction(inst);
      return WalkResult::advance();
    });
  }
}

} // namespace

PassResult OutlineMultipleResults::run(Module *module) {
  auto *structSizeAnalysis =
      module->getOrCreateAnalysis<StructSizeAnalysis>(module);
  const auto &sizeMap = structSizeAnalysis->getStructSizeMap();
  const auto &fieldsMap = structSizeAnalysis->getStructFieldsMap();

  llvm::DenseMap<Function *, llvm::DenseMap<size_t, Phi>> retPhisMap;
  llvm::SmallVector<inst::Call> callsToUpdate;
  llvm::SmallVector<inst::Return> returnsToUpdate;
  for (Function *func : *module->getIR()) {
    FunctionT funcT = func->getFunctionType().cast<FunctionT>();

    auto converted = convertType(funcT, sizeMap);
    if (converted)
      func->setFunctionType(converted->changed);

    if (!func->hasDefinition())
      continue;

    if (converted) {
      IRBuilder builder(module->getContext());
      builder.setInsertionPointStart(func->getEntryBlock());

      llvm::DenseMap<size_t, Phi> retPhis;
      for (const auto [retIdx, _] : converted->convertMap) {
        auto retType = funcT.getReturnTypes()[retIdx];
        auto pointerType = PointerT::get(module->getContext(), retType);
        auto phi = builder.create<Phi>({}, pointerType);
        retPhis[retIdx] = phi;
      }

      retPhisMap[func] = std::move(retPhis);
    }

    func->walk([&](InstructionStorage *inst) -> WalkResult {
      if (auto call = inst->getDefiningInst<inst::Call>())
        callsToUpdate.push_back(call);
      else if (converted) {
        if (auto ret = inst->getDefiningInst<inst::Return>())
          returnsToUpdate.emplace_back(ret);
      }

      return WalkResult::advance();
    });
  }

  for (auto call : callsToUpdate)
    convertCall(call, sizeMap);

  for (auto ret : returnsToUpdate)
    convertRet(ret, retPhisMap[ret.getParentBlock()->getParentFunction()]);

  convertAllValues(module, sizeMap);
  return PassResult::success();
}

} // namespace kecc::ir
