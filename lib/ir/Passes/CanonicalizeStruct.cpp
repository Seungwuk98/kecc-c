#include "kecc/ir/IRAnalyses.h"
#include "kecc/ir/IRAttributes.h"
#include "kecc/ir/IRInstructions.h"
#include "kecc/ir/IRTransforms.h"
#include "kecc/ir/Instruction.h"
#include "kecc/ir/Module.h"
#include "kecc/ir/PatternMatch.h"
#include "kecc/ir/WalkSupport.h"
#include "kecc/utils/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstdint>
#include <numeric>
#include <ranges>

namespace kecc::ir {

// if struct or array size is le than 2 bytes, we can manipulate it as 2
// register values.
//
// Case 1
//  struct X { a : i64, b : i64 }
//
// before:
//
//  allocations:
//    %l0:struct X
//
//  block b0:
//    %b0:i0:struct X = load %l0:struct X*
//    j b1(%b0:i0:struct X)
//
//  block b1:
//    p0:struct X
//    ...
//
// after:
//  allocations:
//    %l0:i64
//    %l1:i64
//
//  block b0:
//    %b0:i0:i64 = load %l0:i0:i64*
//    %b0:i1:i64 = load %l0:i1:i64*
//    j b1(%b0:i0:i64, %b0:i1:i64)
//
//  block b1:
//    %b1:p0:i64
//    %b1:p1:i64
//    ...
//
//
// Case 2
//  struct X { a : i32, b : i32, c : i64 }
//
// before:
//  allocations:
//    %l0:struct X
//
//  block b0:
//    %b0:i0:i32* = getelementptr %l0:struct X* offset 0:i64
//    %b0:i1:i32:a = load %b0:i0:i32*
//    ...
//
// after:
//  allocations:
//    %l0:i64
//    %l1:i64
//
//  block b0:
//    %b0:i0:i64 = load %l0:i0:i64*
//    %b0:i1:i64 = shl %b0:i0:i64 32:i64
//    %b0:i2:i64 = shr %b0:i0:i64 32:i64 %b0:i3:i32 = typecast %b0:i2:i64 to i32
//    ...
//
// if struct or array size is gt than 2 bytes, we should manipulate it as
// pointer
//
// Case 1
//  struct X { a : i64, b : i64, c : i64 }
//
// before:
// fun i32 @func(struct X) {
//  init:
//    bid: b0
//    allocations:
//      %l0:struct X
//
//  block b0:
//    %b0:p0:struct X
//    %b0:i0:unit = store %b0:p0:struct X %l0:struct X*
//    ...
//
//  block b333:
//    ...
//    %b333:i332:struct X = load %l0:struct X*
//    %b333:i333:i32 = call @func(%b333:i332:struct X)
// }
// after:
//
// fun @func(struct X*) {
//  init:
//    bid: b0
//    allocations:
//      %l0:struct X*
//      %l1:struct X
//      %l2:struct X
//
//  block b0:
//    %b0:p0:struct X*
//    %b0:i0:struct X* = load %l0:struct X**
//    %b0:i1:unit = call
//            @memcpy(%l1:struct X*, %b0:i0:struct X*, sizeof(struct X))
//    ...
//
//  block b333:
//    ...
//    %b333:i332:unit = @memcpy(%l2:struct X*, %l0:struct X*, sizeof(struct X))
//    %b333:i333:i32 = call @func(%l2:struct X*)
// }

static size_t findBiggerNrstPow2(size_t value) {
  if (value == 0)
    return 0;
  size_t result = 1;
  while (result < value) {
    result <<= 1;
  }
  return result;
}

class FoldTypeCast : public Pass {
public:
  FoldTypeCast() {}

  PassResult run(Module *module) override;

  llvm::StringRef getPassArgument() const override { return "fold-type-cast"; }

  void setOption(llvm::StringRef option) override {
    if (option == "func-ptr-only")
      funcPtrOnly = true;
    else
      funcPtrOnly = false;
  }

private:
  bool funcPtrOnly = false;
};

class FoldTypeCastPattern : public InstConversionPattern<inst::TypeCast> {
public:
  FoldTypeCastPattern(bool funcPtrOnly) : funcPtrOnly(funcPtrOnly) {}

  utils::LogicalResult matchAndRewrite(IRRewriter &rewriter, Adaptor adaptor,
                                       inst::TypeCast inst) override {
    auto operand = adaptor.getValue();
    auto operandType = operand.getType();
    if (funcPtrOnly && operandType.isa<PointerT>() &&
        !operandType.cast<PointerT>().getPointeeType().isa<FunctionT>())
      return utils::LogicalResult::failure();

    if (operandType == inst.getTargetType()) {
      rewriter.replaceInst(inst.getStorage(), operand);
      return utils::LogicalResult::success();
    }

    return utils::LogicalResult::failure();
  }

private:
  bool funcPtrOnly;
};

PassResult FoldTypeCast::run(Module *module) {
  PatternSet patterns;
  patterns.addPatterns<FoldTypeCastPattern>(funcPtrOnly);

  auto result = applyPatternConversion(module, patterns);
  if (result.isError())
    return PassResult::failure();

  return PassResult::success();
}

class CanonicalizeStructImpl {
public:
  CanonicalizeStructImpl(
      Module *module, StructSizeMap sizeMap, StructFieldsMap fieldsMap,
      llvm::DenseMap<Function *, llvm::DenseMap<size_t, Phi>> retIdx2ArgMap)
      : module(module), sizeMap(std::move(sizeMap)),
        fieldsMap(std::move(fieldsMap)),
        retIdx2ArgMap(std::move(retIdx2ArgMap)) {}

  PassResult run(Function *func);

  void processSmallValue(Value value, Value int64Ptr, Value int64_2Ptr);
  void processBigValue(Value value);
  void processFunctionArg(Phi phi, size_t phiIdx);

  void storeValuesIntoPointer(llvm::SMRange range, IRBuilder &builder,
                              llvm::ArrayRef<Value> values, Value pointer,
                              size_t memorySize, Value int64Ptr,
                              Value int64_2Ptr);

  Value createIntConstant(std::uint64_t value);
  Value createMemCpy();
  Type getVoidPointerT() const {
    return PointerT::get(module->getContext(),
                         UnitT::get(module->getContext()));
  }

private:
  friend class CanonicalizeStruct;
  Type convertFuncT(FunctionT funcT) const;

  Module *module;
  StructSizeMap sizeMap;
  StructFieldsMap fieldsMap;
  llvm::DenseMap<Block *, llvm::SmallVector<int>> originRetIdx;
  llvm::DenseMap<Function *, llvm::DenseMap<size_t, Phi>> retIdx2ArgMap;
  llvm::DenseMap<Value, Value> replaceMap;
};

static std::pair<Type, llvm::SmallVector<size_t>>
convertFuncT(FunctionT funcT, const StructSizeMap &sizeMap) {
  IRContext *context = funcT.getContext();
  llvm::SmallVector<Type> retTypes;
  llvm::SmallVector<Type> argTypes;

  llvm::SmallVector<size_t> converted2Args;
  IntT int64T = IntT::get(context, 64, true);
  for (auto [idx, retT] : llvm::enumerate(funcT.getReturnTypes())) {
    if (auto structT = retT.dyn_cast<NameStruct>()) {
      auto [size, align] = structT.getSizeAndAlign(sizeMap);
      if (size <= 8)
        retTypes.emplace_back(int64T);
      else if (size <= 16) {
        retTypes.emplace_back(int64T);
        retTypes.emplace_back(int64T);
      } else {
        argTypes.insert(argTypes.begin(), PointerT::get(context, structT));
        converted2Args.emplace_back(idx);
      }
    } else
      retTypes.emplace_back(retT);
  }

  for (Type argT : funcT.getArgTypes()) {
    if (auto structT = argT.dyn_cast<NameStruct>()) {
      auto [size, align] = structT.getSizeAndAlign(sizeMap);
      if (size <= 8)
        argTypes.emplace_back(int64T);
      else if (size <= 16) {
        argTypes.emplace_back(int64T);
        argTypes.emplace_back(int64T);
      } else
        argTypes.emplace_back(PointerT::get(context, structT));
    } else
      argTypes.emplace_back(argT);
  }

  if (retTypes.empty())
    retTypes.emplace_back(UnitT::get(context));

  return {FunctionT::get(context, retTypes, argTypes), converted2Args};
}

CanonicalizeStruct::CanonicalizeStruct() {}
CanonicalizeStruct::~CanonicalizeStruct() = default;

void CanonicalizeStruct::init(Module *module) {
  StructSizeAnalysis *structSizeAnalysis =
      module->getAnalysis<StructSizeAnalysis>();
  if (!structSizeAnalysis) {
    auto newAnalysis = StructSizeAnalysis::create(module);
    structSizeAnalysis = newAnalysis.get();
    module->insertAnalysis(std::move(newAnalysis));
  }
  const auto &structSizeMap = structSizeAnalysis->getStructSizeMap();
  const auto &structFieldsMap = structSizeAnalysis->getStructFieldsMap();

  llvm::DenseMap<Function *, llvm::DenseMap<size_t, Phi>> retIdx2ArgMap;
  for (Function *function : *module->getIR()) {
    auto functionT = function->getFunctionType().cast<FunctionT>();
    const auto [replaced, convertedArgs] =
        convertFuncT(functionT, structSizeMap);

    function->setFunctionType(replaced);

    if (functionT != replaced) {
      IRBuilder builder(module->getContext());

      llvm::DenseMap<size_t, Phi> convertedArgsMap;
      for (auto convArg : convertedArgs) {
        Type argT = functionT.getReturnTypes()[convArg];
        Type ptrT = PointerT::get(module->getContext(), argT);

        if (function->hasDefinition()) {
          builder.setInsertionPointStart(function->getEntryBlock());
          auto newArgPhi = builder.create<Phi>({}, ptrT);
          newArgPhi.setValueName("ret_ptr_" + std::to_string(convArg));
          convertedArgsMap[convArg] = newArgPhi;
        }
      }
      retIdx2ArgMap[function] = std::move(convertedArgsMap);
    }
  }

  impl = std::make_unique<CanonicalizeStructImpl>(
      module, std::move(structSizeMap), std::move(structFieldsMap),
      std::move(retIdx2ArgMap));
}
void CanonicalizeStruct::exit(Module *module) { impl.reset(); }

PassResult CanonicalizeStruct::run(Module *module) {
  PassResult retResult = PassResult::skip();
  for (Function *function : *module->getIR()) {
    auto result = impl->run(function);
    if (result.isFailure())
      return result;
    if (result.isSuccess())
      retResult = PassResult::success();
  }

  convertAllFuncT(module);

  return retResult;
}

void CanonicalizeStruct::convertAllFuncT(Module *module) {
  // 1) convert all constant

  auto convertValueT = [&](Value value) {
    auto type = value.getType();
    auto replacedType = type.replace([&](Type type) -> ReplaceResult<Type> {
      if (auto funcT = type.dyn_cast<FunctionT>()) {
        return {impl->convertFuncT(funcT), utils::LogicalResult::success()};
      }
      return {type, utils::LogicalResult::success()};
    });
    if (type != replacedType) {
      value.getImpl()->setType(replacedType);
    }
  };

  auto convertAttr = [&](InstructionStorage *inst) {
    for (auto [idx, attr] : llvm::enumerate(inst->getAttributes())) {
      inst->setAttribute(idx,
                         attr.replace([&](Type type) -> ReplaceResult<Type> {
                           if (auto funcT = type.dyn_cast<FunctionT>()) {
                             return {impl->convertFuncT(funcT),
                                     utils::LogicalResult::success()};
                           }
                           return {type, utils::LogicalResult::success()};
                         }));
    }
  };

  for (InstructionStorage *inst : *module->getIR()->getConstantBlock()) {
    inst::Constant constant = llvm::cast<inst::Constant>(inst);
    convertValueT(constant);
    convertAttr(constant.getStorage());
  }

  // 2) convert all function value
  for (Function *function : *module->getIR()) {
    for (InstructionStorage *inst : *function->getAllocationBlock()) {
      inst::LocalVariable localVar = llvm::cast<inst::LocalVariable>(inst);
      convertValueT(localVar);
      convertAttr(localVar.getStorage());
    }

    function->walk([&](InstructionStorage *storage) -> WalkResult {
      auto results = storage->getResults();
      for (Value value : results)
        convertValueT(value);
      convertAttr(storage);
      return WalkResult::advance();
    });
  }
}

Value CanonicalizeStructImpl::createIntConstant(std::uint64_t value) {
  IRBuilder builder(module->getContext());
  builder.setInsertionPoint(module->getIR()->getConstantBlock());
  return builder.create<inst::Constant>(
      {}, ConstantIntAttr::get(module->getContext(), value, 64, true));
}

Value CanonicalizeStructImpl::createMemCpy() {
  IRBuilder builder(module->getContext());
  builder.setInsertionPoint(module->getIR()->getConstantBlock());

  auto voidPtrT = getVoidPointerT();
  auto int64T = IntT::get(module->getContext(), 64, true);
  auto funcType =
      FunctionT::get(module->getContext(), {UnitT::get(module->getContext())},
                     {voidPtrT, voidPtrT, int64T});
  auto funcPtrT = PointerT::get(module->getContext(), funcType);

  auto func = builder.create<inst::Constant>(
      {}, ConstantVariableAttr::get(module->getContext(), "memcpy", funcPtrT));
  return func;
}

Type CanonicalizeStructImpl::convertFuncT(FunctionT funcT) const {
  return ::kecc::ir::convertFuncT(funcT, sizeMap).first;
}

PassResult CanonicalizeStructImpl::run(Function *func) {
  replaceMap.clear();
  originRetIdx.clear();

  for (Block *block : *func) {
    if (auto ret = block->getExit().dyn_cast<inst::Return>()) {
      auto indices = std::views::iota(0, (int)ret.getValues().size());
      llvm::SmallVector<int> retIdx(indices.begin(), indices.end());
      originRetIdx[block] = std::move(retIdx);
    }
  }

  IRBuilder builder(module->getContext());
  builder.setInsertionPoint(func->getAllocationBlock());

  auto int64T = IntT::get(module->getContext(), 64, true);
  auto arrayInt64T = ArrayT::get(module->getContext(), 2, int64T);
  auto int64Ptr = builder.create<inst::LocalVariable>(
      {}, PointerT::get(builder.getContext(), int64T));
  auto int64_2Ptr = builder.create<inst::LocalVariable>(
      {}, PointerT::get(builder.getContext(), arrayInt64T));

  llvm::SmallVector<Value> smallStructs;
  llvm::SmallVector<Value> bigStructs;
  func->walk([&](InstructionStorage *storage) -> WalkResult {
    auto results = storage->getResults();
    for (Value value : results) {
      if (auto structT = value.getType().dyn_cast<NameStruct>()) {
        auto [size, align] = structT.getSizeAndAlign(sizeMap);
        if (size <= 16)
          smallStructs.emplace_back(value);
        else
          bigStructs.emplace_back(value);
      }
    }
    return WalkResult::advance();
  });

  for (Value value : smallStructs) {
    while (replaceMap.contains(value))
      value = replaceMap.at(value);

    processSmallValue(value, int64Ptr, int64_2Ptr);
  }

  if (!int64Ptr.getResult().hasUses())
    module->removeInst(int64Ptr.getStorage());

  if (!int64_2Ptr.getResult().hasUses())
    module->removeInst(int64_2Ptr.getStorage());

  for (Value value : bigStructs) {
    while (replaceMap.contains(value))
      value = replaceMap.at(value);

    processBigValue(value);
  }

  return smallStructs.empty() && bigStructs.empty() ? PassResult::skip()
                                                    : PassResult::success();
}

void CanonicalizeStructImpl::processSmallValue(Value value, Value int64Ptr,
                                               Value int64_2Ptr) {
  assert(replaceMap.find(value) == replaceMap.end() &&
         "Value must not be already replaced");

  auto inst = value.getInstruction();
  auto block = inst->getParentBlock();
  auto structName = value.getType().cast<NameStruct>().getName();
  const auto [size, align, offsets] = sizeMap.at(structName);
  llvm::ArrayRef<Type> fields = fieldsMap.at(structName);

  // if struct size is le than 1 bytes -> convert to i64
  IRBuilder builder(module->getContext());
  builder.setInsertionPointAfterInst(inst);

  auto i64T = IntT::get(module->getContext(), 64, true);

  llvm::SmallVector<Value> newValues;
  // load, call, phi

  if (auto phi = inst->getDefiningInst<Phi>()) {
    if (size <= 8) {
      auto newPhi = builder.create<Phi>(phi.getRange(), i64T);
      newPhi.setValueName(value.getValueName());

      newValues.emplace_back(newPhi);
    } else {
      auto newPhi = builder.create<Phi>(phi.getRange(), i64T);
      newPhi.setValueName(value.getValueName());
      auto newPhi2 = builder.create<Phi>(phi.getRange(), i64T);
      newPhi2.setValueName(value.getValueName());

      newValues.emplace_back(newPhi);
      newValues.emplace_back(newPhi2);
    }
  } else if (auto call = inst->getDefiningInst<inst::Call>()) {
    auto func = call.getFunction();
    auto functionT =
        func.getType().cast<PointerT>().getPointeeType().cast<FunctionT>();

    auto returnTypes = functionT.getReturnTypes();

    llvm::SmallVector<Type> newReturnTypes;
    newReturnTypes.reserve(returnTypes.size() + (size > 8));

    for (auto idx = 0u; idx < returnTypes.size(); ++idx) {
      if (idx == value.getImpl()->getValueNumber()) {
        auto int64T = IntT::get(module->getContext(), 64, true);
        newReturnTypes.emplace_back(int64T);
        if (size > 8)
          newReturnTypes.emplace_back(int64T);
      } else {
        newReturnTypes.emplace_back(returnTypes[idx]);
      }
    }

    auto newFuncType = FunctionT::get(module->getContext(), newReturnTypes,
                                      functionT.getArgTypes());

    // this cast instruction must be removed later
    auto newFunc = builder.create<inst::TypeCast>(
        call.getRange(), func,
        PointerT::get(module->getContext(), newFuncType));

    auto newCall = builder.create<inst::Call>(
        call.getRange(), newFunc,
        llvm::map_to_vector(call.getArguments(),
                            [&](Value arg) { return arg; }));

    for (unsigned i = 0u, newi = 0u, e = call->getResultSize(); i < e;
         ++i, ++newi) {
      auto oldValue = call.getResult(i);
      if (i == value.getImpl()->getValueNumber()) {
        auto newValue = newCall.getResult(newi);
        newValue.getImpl()->setValueName(oldValue.getValueName());
        newValues.emplace_back(newValue);
        if (size > 8) {
          newValue = newCall.getResult(++newi);
          newValues.emplace_back(newValue);
          newValue.getImpl()->setValueName(oldValue.getValueName());
        }
      } else {
        auto newValue = newCall.getResult(newi);
        newValue.getImpl()->setValueName(oldValue.getValueName());
        oldValue.replaceWith(newValue);
        replaceMap[oldValue] = newValue;
      }
    }

  } else if (auto load = inst->getDefiningInst<inst::Load>()) {
    auto pointer = load.getPointer();
    auto pointerType = pointer.getType().cast<PointerT>();
    auto pointeeType = pointerType.getPointeeType();
    assert(pointeeType == value.getType() && "Must be same struct type");

    // if struct size is le than 8 bytes
    //   1) size == 2^x
    //      -> cast pointer to correspond integer type
    //      -> load value and cast to i64 if needed
    //   2) else
    //      -> memcpy to int64Ptr
    //      -> load value from int64Ptr
    //
    // if struct size is 16 bytes
    //   -> cast pointer to i64 ptr
    //   -> load first value
    //   -> getelementptr to second i64 ptr
    //   -> load values from second i64 ptr
    //
    // if o.w
    //   -> memcpy to int64_2Ptr
    //   -> load first value from int64_2Ptr
    //   -> getelementptr to second i64 ptr
    //   -> load values from second i64 ptr

    auto pow2 = findBiggerNrstPow2(size);
    if (size <= 8 && size == pow2) {
      auto correspondT = IntT::get(module->getContext(), size * 8, true);
      auto correspondPointerT =
          PointerT::get(module->getContext(), correspondT);
      pointer = builder.create<inst::TypeCast>(load.getRange(), pointer,
                                               correspondPointerT);
      Value newLoad = builder.create<inst::Load>(load.getRange(), pointer);
      if (newLoad.getType() != i64T) {
        newLoad =
            builder.create<inst::TypeCast>(load.getRange(), newLoad, i64T);
      }
      newValues.emplace_back(newLoad);
    } else if (size <= 8) {
      auto memcpy = createMemCpy();
      auto constMemSize = createIntConstant(size);
      auto voidPtrT = getVoidPointerT();
      auto voidInt64Ptr =
          builder.create<inst::TypeCast>(load.getRange(), int64Ptr, voidPtrT);
      if (pointer.getType() != voidPtrT)
        pointer =
            builder.create<inst::TypeCast>(load.getRange(), pointer, voidPtrT);

      builder.create<inst::Call>(
          load.getRange(), memcpy,
          llvm::SmallVector<Value>{voidInt64Ptr, pointer, constMemSize});

      auto newLoad = builder.create<inst::Load>(load.getRange(), int64Ptr);
      newValues.emplace_back(newLoad);
    } else if (size == 16) {
      auto int64PtrT = PointerT::get(module->getContext(),
                                     IntT::get(module->getContext(), 64, true));
      auto ptr0 =
          builder.create<inst::TypeCast>(load.getRange(), pointer, int64PtrT);
      auto newLoad0 = builder.create<inst::Load>(load.getRange(), ptr0);
      newValues.emplace_back(newLoad0);

      auto const8 = createIntConstant(8);
      auto ptr1 = builder.create<inst::Gep>(load.getRange(), pointer, const8,
                                            int64PtrT);
      auto newLoad1 = builder.create<inst::Load>(load.getRange(), ptr1);
      newValues.emplace_back(newLoad1);
    } else {
      auto memcpy = createMemCpy();
      auto constMemSize = createIntConstant(size);
      auto voidPtrT = getVoidPointerT();
      auto voidInt64_2Ptr =
          builder.create<inst::TypeCast>(load.getRange(), int64_2Ptr, voidPtrT);

      pointer =
          builder.create<inst::TypeCast>(load.getRange(), pointer, voidPtrT);

      builder.create<inst::Call>(
          load.getRange(), memcpy,
          llvm::SmallVector<Value>{voidInt64_2Ptr, pointer, constMemSize});

      auto ptr0 = builder.create<inst::TypeCast>(
          load->getRange(), int64_2Ptr,
          PointerT::get(builder.getContext(), i64T));
      auto newLoad0 = builder.create<inst::Load>(load.getRange(), ptr0);
      newValues.emplace_back(newLoad0);
      auto const8 = createIntConstant(8);
      auto ptr1 =
          builder.create<inst::Gep>(load.getRange(), int64_2Ptr, const8,
                                    PointerT::get(module->getContext(), i64T));
      auto newLoad1 = builder.create<inst::Load>(load.getRange(), ptr1);
      newValues.emplace_back(newLoad1);
    }
  } else if (auto constant = value.getDefiningInst<inst::OutlineConstant>()) {
    auto inlineConstant =
        constant.getConstant().getDefiningInst<ir::inst::Constant>();
    assert(inlineConstant && "Expected constant value");
    auto constantValue = inlineConstant.getValue();
    assert(constantValue.isa<ir::ConstantUndefAttr>() && "Expected undef");

    IRBuilder::InsertionGuard guard(builder);

    builder.setInsertionPoint(module->getIR()->getConstantBlock());
    auto undefI64 = builder.create<ir::inst::Constant>(
        {}, ir::ConstantUndefAttr::get(module->getContext(), i64T));

    newValues.emplace_back(undefI64);
    if (size > 8)
      newValues.emplace_back(undefI64);
  } else {
    llvm_unreachable("Unexpected instruction type for struct manipulation.");
  }

  llvm::DenseSet<InstructionStorage *> valueUsers;
  for (Operand *operand : *value.getImpl()) {
    InstructionStorage *user = operand->getOwner();
    valueUsers.insert(user);
  }

  for (InstructionStorage *user : valueUsers) {
    // value users
    // 1) function call
    //    @func(..., struct X, ...) -> @func(..., i64, i64, ...)
    // 2) store
    //    store values into pointer
    // 3) phi arg in block exit
    //    bX(V:sturct X) -> bX(v0:i64, v1:i64)

    builder.setInsertionPointBeforeInst(user);
    if (auto call = user->getDefiningInst<inst::Call>()) {
      auto func = call.getFunction();

      llvm::DenseSet<unsigned> oldValueIdx;
      for (unsigned i = 0u, e = call.getArguments().size(); i < e; ++i) {
        if (call.getArguments()[i] == value)
          oldValueIdx.insert(i);
      }

      llvm::SmallVector<Value> newArgs;
      newArgs.reserve(call.getArguments().size() + (newValues.size() > 1));

      auto int64T = IntT::get(module->getContext(), 64, true);
      for (unsigned i = 0u; i < call.getArguments().size(); ++i) {
        if (oldValueIdx.contains(i)) {
          newArgs.emplace_back(newValues[0]);
          if (newValues.size() > 1)
            newArgs.emplace_back(newValues[1]);
        } else
          newArgs.emplace_back(call.getArguments()[i]);
      }

      auto newArgTypes =
          llvm::map_to_vector(newArgs, [](Value arg) { return arg.getType(); });

      auto retTypes = llvm::map_to_vector(
          call->getResults(), [](Value res) { return res.getType(); });

      auto newFuncType =
          FunctionT::get(module->getContext(), retTypes, newArgTypes);

      // this cast instruction must be removed later
      auto newFunc = builder.create<inst::TypeCast>(
          call.getRange(), func,
          PointerT::get(module->getContext(), newFuncType));
      auto newCall =
          builder.create<inst::Call>(call.getRange(), newFunc, newArgs);

      // update replace map
      for (auto [from, to] :
           llvm::zip(call->getResults(), newCall->getResults()))
        replaceMap[from] = to;

      module->replaceInst(user, newCall->getResults(), true);
    } else if (auto store = user->getDefiningInst<inst::Store>()) {
      storeValuesIntoPointer(store.getRange(), builder, newValues,
                             store.getPointer(), size, int64Ptr, int64_2Ptr);
      module->removeInst(user);
    } else if (auto ret = user->getDefiningInst<inst::Return>()) {
      auto oldValues = ret.getValues();
      assert(oldValues.size() == originRetIdx[block].size() &&
             "Return values size must match the original return index size");
      llvm::SmallVector<Value> newRetValues;
      newRetValues.reserve(oldValues.size() + (newValues.size() > 1));
      llvm::SmallVector<int> &retIdx = originRetIdx[block];

      for (unsigned i = 0u; i < oldValues.size(); ++i) {
        if (oldValues[i] == value) {
          newRetValues.emplace_back(newValues[0]);
          retIdx[i] = -1;
          if (newValues.size() > 1) {
            newRetValues.emplace_back(newValues[1]);
            retIdx.insert(retIdx.begin() + i, -1);
          }
        } else
          newRetValues.emplace_back(oldValues[i]);
      }

      auto newRet = builder.create<inst::Return>(ret.getRange(), newRetValues);
      module->replaceInst(user, newRet->getResults(), true);
    } else if (auto blockExit = user->getDefiningInst<BlockExit>()) {
      assert(llvm::all_of(blockExit->getOperands(),
                          [&](const Operand &op) { return op != value; }) &&
             "struct value cannont be operand of block exit");

      // no need to use `Module::replaceExit` because we are not changing the
      // relationship between blocks
      for (auto [idx, jumpArg] : llvm::enumerate(blockExit->getJumpArgs())) {
        JumpArgState jumpArgState = jumpArg->getAsState();
        JumpArgState newJumpArgState(jumpArgState.getBlock());
        for (auto arg : jumpArgState.getArgs()) {
          if (arg == value) {
            newJumpArgState.pushArg(newValues[0]);
            if (newValues.size() > 1)
              newJumpArgState.pushArg(newValues[1]);
          } else
            newJumpArgState.pushArg(arg);
        }
        blockExit->setJumpArg(idx, newJumpArgState);
      }
    }
  }
  // finally remove the instruction which is declaring the struct value
  module->removeInst(inst);
}

void CanonicalizeStructImpl::storeValuesIntoPointer(
    llvm::SMRange range, IRBuilder &builder, llvm::ArrayRef<Value> values,
    Value pointer, size_t memorySize, Value int64Ptr, Value int64_2Ptr) {
  assert(llvm::all_of(values,
                      [&](Value v) {
                        return v.getType().getSizeAndAlign(sizeMap).first == 8;
                      }) &&
         "All values must be 8 bytes in size for this operation");
  assert(values.size() <= 2 &&
         "This function only supports up to 2 values for now");

  auto totalValueSize = values.size() * 8;

  if (totalValueSize == memorySize) {
    auto value0 = values[0];

    auto ptr0 = pointer;
    if (value0.getType() !=
        pointer.getType().cast<PointerT>().getPointeeType()) {
      ptr0 = builder.create<inst::TypeCast>(
          range, pointer,
          PointerT::get(module->getContext(), value0.getType()));
    }
    auto storeInst = builder.create<inst::Store>(range, value0, ptr0);

    if (values.size() > 1) {
      auto value1 = values[1];
      auto const8 = createIntConstant(8);
      auto ptr1 = builder.create<inst::Gep>(
          range, pointer, const8,
          PointerT::get(module->getContext(), value1.getType()));
      storeInst = builder.create<inst::Store>(range, value1, ptr1);
    }
  } else if (totalValueSize > memorySize) {
    // 1) values.size() == 1 && memorySize = 2^x
    //   -> (1) truc value[0] to memrySize
    //   -> (2) store truced value[0] into memory
    // 2) memcpy
    //   -> values.size() == 1
    //      -> (1) store value[0] into int64Ptr
    //      -> (2) memcpy int64Ptr to memory
    //   -> values.size() == 2
    //      -> (1) store value[0] into int64_2Ptr
    //      -> (2) store value[1] into int64_2Ptr + 8
    //      -> (3) memcpy int64_2Ptr to memory

    auto pow2Size = findBiggerNrstPow2(memorySize);
    if (values.size() == 1 && pow2Size == memorySize) {
      assert(memorySize <= 8);
      auto value0 = values[0];

      if (value0.getType().getSizeAndAlign(sizeMap).first != memorySize) {
        value0 = builder.create<inst::TypeCast>(
            range, value0,
            IntT::get(module->getContext(), memorySize * 8, true));
      }

      if (value0.getType() !=
          pointer.getType().cast<PointerT>().getPointeeType()) {
        pointer = builder.create<inst::TypeCast>(
            range, pointer,
            PointerT::get(module->getContext(), value0.getType()));
      }

      auto storeInst = builder.create<inst::Store>(range, value0, pointer);
    } else if (values.size() == 1) {
      auto value0 = values[0];
      auto store0 = builder.create<inst::Store>(range, value0, int64Ptr);

      auto memcpy = createMemCpy();
      auto constMemSize = createIntConstant(memorySize);
      auto voidPtrT = getVoidPointerT();

      if (pointer.getType() != voidPtrT)
        pointer = builder.create<inst::TypeCast>(range, pointer, voidPtrT);

      auto voidInt64Ptr =
          builder.create<inst::TypeCast>(range, int64Ptr, voidPtrT);

      builder.create<inst::Call>(
          range, memcpy,
          llvm::SmallVector<Value>{pointer, voidInt64Ptr, constMemSize});
    } else {
      auto value0 = values[0];
      auto value1 = values[1];

      auto castedInt64_2Ptr = builder.create<inst::TypeCast>(
          range, int64_2Ptr,
          PointerT::get(module->getContext(),
                        IntT::get(module->getContext(), 64, true)));
      auto store0 =
          builder.create<inst::Store>(range, value0, castedInt64_2Ptr);
      auto const8 = createIntConstant(8);
      auto pointer1 = builder.create<inst::Gep>(
          range, int64_2Ptr, const8,
          PointerT::get(module->getContext(), value1.getType()));

      auto store1 = builder.create<inst::Store>(range, value1, pointer1);

      auto memcpy = createMemCpy();
      auto constMemSize = createIntConstant(memorySize);
      auto voidPtrT = getVoidPointerT();
      if (pointer.getType() != voidPtrT)
        pointer = builder.create<inst::TypeCast>(range, pointer, voidPtrT);
      auto voidInt64_2Ptr =
          builder.create<inst::TypeCast>(range, int64_2Ptr, voidPtrT);

      builder.create<inst::Call>(
          range, memcpy,
          llvm::SmallVector<Value>{pointer, voidInt64_2Ptr, constMemSize});
    }
  } else
    llvm_unreachable("Unexpected total value size for struct manipulation."
                     "Because total value size is always bigger than memory.");
}

void CanonicalizeStructImpl::processBigValue(Value value) {
  assert(replaceMap.find(value) == replaceMap.end() &&
         "Value must not be already replaced");

  auto *inst = value.getInstruction();
  auto structT = value.getType().cast<NameStruct>();

  Value storedPointer;
  if (auto phi = inst->getDefiningInst<Phi>()) {
    IRBuilder builder(module->getContext());
    builder.setInsertionPointAfterInst(inst);

    auto phiT = phi.getType();
    auto phiPointerT = PointerT::get(module->getContext(), phiT);

    auto newPhi = builder.create<Phi>(phi.getRange(), phiPointerT);
    newPhi.setValueName(phi.getValueName());

    storedPointer = newPhi;
  } else if (auto call = inst->getDefiningInst<inst::Call>()) {
    // Create new pointer and give it to the function
    // The function must be changed to accept a pointer to the struct instead of
    // the struct itself

    llvm::SmallVector<Type> newReturnTypes;
    newReturnTypes.reserve(call->getResultSize() - 1);

#ifndef NDEBUG
    /// Calling this function must be ordered.
    ///
    /// v0:struct X, v1:struct Y = call @func()
    /// -> unit = call @func(struct Y*, struct X*)
    bool found = false;
    for (unsigned i = 0u; i < call->getResultSize(); ++i) {
      if (call.getResult(i) == value) {
        found = true;
      } else {
        newReturnTypes.emplace_back(call.getResult(i).getType());
        if (!found && call.getResult(i).getType().isa<NameStruct>()) {
          assert(call.getResult(i)
                         .getType()
                         .cast<NameStruct>()
                         .getSizeAndAlign(sizeMap)
                         .first <= 16 &&
                 "Struct size must be le than 16 bytes for this operation");
        }
      }
    }
#else
    for (unisgned i = 0u; i < call->getResultSize(); ++i) {
      if (call.getResult(i) != value)
        newReturnTypes.emplace_back(call.getResult(i).getType());
    }
#endif

    if (newReturnTypes.empty())
      newReturnTypes.emplace_back(UnitT::get(module->getContext()));

    IRBuilder builder(module->getContext());
    // create temporary allocation
    builder.setInsertionPoint(
        inst->getParentBlock()->getParentFunction()->getAllocationBlock());
    auto structValuePtr = builder.create<inst::LocalVariable>(
        {}, PointerT::get(builder.getContext(), value.getType()));

    llvm::SmallVector<Value> newArgs;
    newArgs.reserve(call.getArguments().size() + 1);
    newArgs.emplace_back(structValuePtr);

    newArgs.append(call.getArguments().begin(), call.getArguments().end());

    const llvm::SmallVector<Type> newArgTypes =
        llvm::map_to_vector(newArgs, [](Value v) { return v.getType(); });

    auto newFuncT =
        FunctionT::get(module->getContext(), newReturnTypes, newArgTypes);

    builder.setInsertionPointBeforeInst(inst);
    auto newFunc = builder.create<inst::TypeCast>(
        call.getRange(), call.getFunction(),
        PointerT::get(module->getContext(), newFuncT));

    auto newCall =
        builder.create<inst::Call>(call.getRange(), newFunc, newArgs);

    for (auto idx = 0u, newi = 0u; idx < call->getResultSize(); ++idx) {
      auto oldValue = call.getResult(idx);
      if (oldValue != value) {
        auto newValue = newCall.getResult(newi++);
        newValue.getImpl()->setValueName(oldValue.getValueName());
        oldValue.replaceWith(newValue);
        replaceMap[oldValue] = newValue;
      }
    }

    storedPointer = structValuePtr;
  } else if (auto load = inst->getDefiningInst<inst::Load>()) {
    // create new pointer to the struct value and memcpy
    IRBuilder builder(module->getContext());
    builder.setInsertionPoint(
        inst->getParentBlock()->getParentFunction()->getAllocationBlock());

    auto structValuePtr = builder.create<inst::LocalVariable>(
        {}, PointerT::get(builder.getContext(), value.getType()));

    builder.setInsertionPointBeforeInst(inst);
    auto memcpy = createMemCpy();
    auto constMemSize =
        createIntConstant(structT.getSizeAndAlign(sizeMap).first);

    auto voidPtrT = getVoidPointerT();
    auto pointer = load.getPointer();
    auto voidStructPtr = builder.create<inst::TypeCast>(
        load.getRange(), structValuePtr, voidPtrT);
    if (pointer.getType() != voidPtrT)
      pointer =
          builder.create<inst::TypeCast>(load.getRange(), pointer, voidPtrT);

    builder.create<inst::Call>(
        load.getRange(), memcpy,
        llvm::SmallVector<Value>{voidStructPtr, pointer, constMemSize});
    storedPointer = structValuePtr;
  } else if (auto outlineConstant =
                 inst->getDefiningInst<inst::OutlineConstant>()) {
    IRBuilder builder(module->getContext());
    builder.setInsertionPoint(
        inst->getParentBlock()->getParentFunction()->getAllocationBlock());

    storedPointer = builder.create<inst::LocalVariable>(
        {}, PointerT::get(builder.getContext(), value.getType()));

  } else
    llvm_unreachable("Unexpected instruction type for struct manipulation.");

  llvm::SmallVector<InstructionStorage *> users;
  for (Operand *operand : *value.getImpl()) {
    auto user = operand->getOwner();
    users.emplace_back(user);
  }

  for (InstructionStorage *user : users) {
    if (auto store = user->getDefiningInst<inst::Store>()) {
      // memcpy values into pointer

      IRBuilder builder(module->getContext());
      builder.setInsertionPointAfterInst(store.getStorage());
      auto pointer = store.getPointer();

      auto voidPtrT = getVoidPointerT();
      if (pointer.getType() != voidPtrT)
        pointer =
            builder.create<inst::TypeCast>(store.getRange(), pointer, voidPtrT);

      auto constMemSize =
          createIntConstant(structT.getSizeAndAlign(sizeMap).first);

      auto voidStoredPointer = builder.create<inst::TypeCast>(
          store.getRange(), storedPointer, voidPtrT);
      auto memcpy = createMemCpy();
      builder.create<inst::Call>(
          store.getRange(), memcpy,
          llvm::SmallVector<Value>{pointer, voidStoredPointer, constMemSize});

      module->removeInst(user);
    } else if (auto call = user->getDefiningInst<inst::Call>()) {
      // replace the argument to pointer
      // The function must be changed to accept a pointer to the struct

      llvm::SmallVector<Value> newArgs;
      newArgs.reserve(call.getArguments().size());

      for (unsigned i = 0u; i < call.getArguments().size(); ++i) {
        if (call.getArguments()[i] == value) {
          newArgs.emplace_back(storedPointer);
        } else {
          newArgs.emplace_back(call.getArguments()[i]);
        }
      }

      llvm::SmallVector<Type> newTypes =
          llvm::map_to_vector(newArgs, [](Value v) { return v.getType(); });

      auto func = call.getFunction();
      auto functionT =
          func.getType().cast<PointerT>().getPointeeType().cast<FunctionT>();
      auto returnTypes = functionT.getReturnTypes();
      auto newFuncType =
          FunctionT::get(module->getContext(), returnTypes, newTypes);

      // cast function
      IRBuilder builder(module->getContext());
      builder.setInsertionPointBeforeInst(user);

      auto newFunc = builder.create<inst::TypeCast>(
          call.getRange(), func,
          PointerT::get(module->getContext(), newFuncType));
      auto newCall =
          builder.create<inst::Call>(call.getRange(), newFunc, newArgs);

      // update replace map
      for (auto [from, to] :
           llvm::zip(call->getResults(), newCall->getResults()))
        replaceMap[from] = to;

      module->replaceInst(user, newCall->getResults(), true);
    } else if (auto ret = user->getDefiningInst<inst::Return>()) {
      // replace the return value to pointer

      auto operands = ret->getOperands();
      auto idx = llvm::find(operands, value) - operands.begin();
      llvm::SmallVector<int> &originIndices =
          originRetIdx[user->getParentBlock()];
      assert(operands.size() == originIndices.size() &&
             "Return values must have the same size as origin indices");
      auto originIdx = originIndices[idx];

      auto destPtr =
          retIdx2ArgMap[user->getParentBlock()->getParentFunction()][originIdx];
      assert(destPtr && "Struct value must be mapped to a function argument");

      IRBuilder builder(module->getContext());
      builder.setInsertionPointBeforeInst(user);

      auto memcpy = createMemCpy();
      auto constMemSize =
          createIntConstant(structT.getSizeAndAlign(sizeMap).first);
      auto voidPtrT = getVoidPointerT();
      auto voidDestPtr =
          builder.create<inst::TypeCast>(ret.getRange(), destPtr, voidPtrT);
      auto voidSrcPtr = builder.create<inst::TypeCast>(ret.getRange(),
                                                       storedPointer, voidPtrT);

      builder.create<inst::Call>(
          ret.getRange(), memcpy,
          llvm::SmallVector<Value>{voidDestPtr, voidSrcPtr, constMemSize});

      llvm::SmallVector<Value> newRetValues;
      newRetValues.reserve(ret.getValues().size() - 1);

      for (unsigned i = 0u; i < ret.getValues().size(); ++i) {
        if (ret.getValues()[i] != value)
          newRetValues.emplace_back(ret.getValues()[i]);
        else
          originIndices.erase(originIndices.begin() + i);
      }

      builder.create<inst::Return>(ret.getRange(), newRetValues);
      module->removeInst(user);

    } else if (auto blockExit = user->getDefiningInst<BlockExit>()) {
      // replace the jump argument to pointer
      assert(llvm::all_of(blockExit->getOperands(),
                          [&](const Operand &op) { return op != value; }) &&
             "struct value cannont be operand of block exit");

      // no need to use `Module::replaceExit` because we are not changing the
      // relationship between blocks
      for (auto [idx, jumpArg] : llvm::enumerate(blockExit->getJumpArgs())) {
        JumpArgState jumpArgState = jumpArg->getAsState();
        JumpArgState newJumpArgState(jumpArgState.getBlock());
        for (auto arg : jumpArgState.getArgs()) {
          if (arg == value)
            newJumpArgState.pushArg(storedPointer);
          else
            newJumpArgState.pushArg(arg);
        }
        blockExit->setJumpArg(idx, newJumpArgState);
      }
    }
  }

  module->removeInst(inst);
}

void registerCanonicalizeStructPasses() {
  registerPass<CanonicalizeStruct>();
  registerPass<FoldTypeCast>();
}

} // namespace kecc::ir
