#include "kecc/ir/IRInstructions.h"
#include "kecc/ir/IRTransforms.h"
#include "kecc/ir/Instruction.h"
#include "kecc/ir/Module.h"
#include "kecc/ir/PatternMatch.h"
#include "kecc/ir/WalkSupport.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstdint>

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

class CanonicalizeStructImpl {
public:
  CanonicalizeStructImpl(Module *module, StructSizeMap sizeMap,
                         StructFieldsMap fieldsMap)
      : module(module), sizeMap(std::move(sizeMap)),
        fieldsMap(std::move(fieldsMap)) {}

  PassResult run(Function *func);

  void processSmallValue(Value value, Value int64Ptr, Value int64_2Ptr);
  void processBigValue(Value value);
  void processFunctionArg(Phi phi, size_t phiIdx);

  void storeValuesIntoPointer(llvm::SMRange range, IRBuilder &builder,
                              llvm::ArrayRef<Value> values, Value pointer,
                              size_t memorySize, Value int64Ptr,
                              Value int64_2Ptr);

  Value createIntConstant(std::uint64_t value, Type intT);
  Value createMemCpy(llvm::SMRange range);
  Type getVoidPointerT() const {
    return PointerT::get(module->getContext(),
                         UnitT::get(module->getContext()));
  }

private:
  Module *module;
  StructSizeMap sizeMap;
  StructFieldsMap fieldsMap;
};

void CanonicalizeStruct::init(Module *module) {
  auto [structSizeMap, structFieldsMap] = module->calcStructSizeMap();
  impl = std::make_unique<CanonicalizeStructImpl>(
      module, std::move(structSizeMap), std::move(structFieldsMap));
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

  return retResult;
}

PassResult CanonicalizeStructImpl::run(Function *func) {
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
        if (size <= 2)
          smallStructs.emplace_back(value);
        else
          bigStructs.emplace_back(value);
      }
    }
    return WalkResult::advance();
  });

  for (Value value : smallStructs)
    processSmallValue(value, int64Ptr, int64_2Ptr);

  if (!int64Ptr.getResult().hasUses())
    module->removeInst(int64Ptr.getStorage());

  if (!int64_2Ptr.getResult().hasUses())
    module->removeInst(int64_2Ptr.getStorage());

  for (Value value : bigStructs)
    processBigValue(value);

  return smallStructs.empty() && bigStructs.empty() ? PassResult::skip()
                                                    : PassResult::success();
}

void CanonicalizeStructImpl::processSmallValue(Value value, Value int64Ptr,
                                               Value int64_2Ptr) {
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
      auto newPhi = builder.create<Phi>(phi.getRange(), structName);
      newPhi.setValueName(value.getValueName());
      auto newPhi2 = builder.create<Phi>(phi.getRange(), structName);
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

    auto newCall = builder.create<inst::Call>(call.getRange(), newFunc,
                                              call.getArguments());

    for (unsigned i = 0u, newi = 0u, e = call->getResultSize(); i < e;
         ++i, ++newi) {
      if (i == value.getImpl()->getValueNumber()) {
        auto newValue = newCall.getResult(newi);
        newValue.getImpl()->setValueName(call.getResult(i).getValueName());
        newValues.emplace_back(newValue);
        if (size > 8) {
          newValue = newCall.getResult(++newi);
          newValues.emplace_back(newValue);
          newValue.getImpl()->setValueName(call.getResult(i).getValueName());
        }
      } else {
        auto newValue = newCall.getResult(newi);
        newValue.getImpl()->setValueName(call.getResult(i).getValueName());
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
    // if struct size is gt than 8 bytes
    //   -> memcpy to int64_2Ptr
    //   -> load values from int64_2Ptr

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
      auto memcpy = createMemCpy(load.getRange());
      auto constMemSize =
          createIntConstant(size, IntT::get(module->getContext(), 64, true));
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
    } else {
      auto memcpy = createMemCpy(load.getRange());
      auto constMemSize =
          createIntConstant(size, IntT::get(module->getContext(), 64, true));
      auto voidPtrT = getVoidPointerT();
      auto voidInt64_2Ptr =
          builder.create<inst::TypeCast>(load.getRange(), int64_2Ptr, voidPtrT);
      if (pointer.getType() != voidPtrT)
        pointer =
            builder.create<inst::TypeCast>(load.getRange(), pointer, voidPtrT);
      builder.create<inst::Call>(
          load.getRange(), memcpy,
          llvm::SmallVector<Value>{voidInt64_2Ptr, pointer, constMemSize});

      auto newLoad0 = builder.create<inst::Load>(load.getRange(), int64_2Ptr);
      newValues.emplace_back(newLoad0);
      auto const8 =
          createIntConstant(8, IntT::get(module->getContext(), 64, true));
      auto pointer1 =
          builder.create<inst::Gep>(load.getRange(), int64_2Ptr, const8,
                                    PointerT::get(module->getContext(), i64T));
      auto newLoad1 = builder.create<inst::Load>(load.getRange(), pointer1);
      newValues.emplace_back(newLoad1);
    }
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

    if (auto call = user->getDefiningInst<inst::Call>()) {
      auto func = call.getFunction();
      auto functionT =
          func.getType().cast<PointerT>().getPointeeType().cast<FunctionT>();

      llvm::DenseSet<unsigned> oldValueIdx;
      for (unsigned i = 0u, e = call->getResultSize(); i < e; ++i) {
        if (call.getResult(i) == value) {
          oldValueIdx.insert(i);
        }
      }

      auto argTypes = functionT.getArgTypes();
      llvm::SmallVector<Type> newArgTypes;
      newArgTypes.reserve(argTypes.size() + (newValues.size() > 1));
      llvm::SmallVector<Value> newArgs;
      newArgs.reserve(argTypes.size() + (newValues.size() > 1));

      auto int64T = IntT::get(module->getContext(), 64, true);
      for (unsigned i = 0u; i < argTypes.size(); ++i) {
        if (oldValueIdx.contains(i)) {
          newArgTypes.emplace_back(int64T);
          newArgs.emplace_back(newValues[0]);
          if (newValues.size() > 1) {
            newArgTypes.emplace_back(int64T);
            newArgs.emplace_back(newValues[1]);
          }
        } else
          newArgTypes.emplace_back(argTypes[i]);
      }

      auto newFuncType = FunctionT::get(module->getContext(), newArgTypes,
                                        functionT.getReturnTypes());

      // this cast instruction must be removed later
      auto newFunc = builder.create<inst::TypeCast>(
          call.getRange(), func,
          PointerT::get(module->getContext(), newFuncType));
      auto newCall =
          builder.create<inst::Call>(call.getRange(), newFunc, newArgs);
      module->replaceInst(user, newCall.getStorage());
    } else if (auto store = user->getDefiningInst<inst::Store>()) {
      storeValuesIntoPointer(store.getRange(), builder, newValues,
                             store.getPointer(), size, int64Ptr, int64_2Ptr);
    } else if (auto blockExit = user->getDefiningInst<BlockExit>()) {
      assert(llvm::all_of(blockExit->getOperands(),
                          [&](const Operand &op) { return op != value; }) &&
             "struct value cannont be operand of block exit");

      // no need to use `Module::replaceExit` because we are not changing the
      // relationship between blocks
      for (auto [idx, jumpArg] : llvm::enumerate(blockExit->getJumpArgs())) {
        JumpArgState jumpArgState = jumpArg->getAsState();
        JumpArgState newJumpArgState(jumpArgState.getBlock());
        for (auto [argIdx, arg] : llvm::enumerate(jumpArgState.getArgs())) {
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
    if (value0.getType() !=
        pointer.getType().cast<PointerT>().getPointeeType()) {
      pointer = builder.create<inst::TypeCast>(
          range, pointer,
          PointerT::get(module->getContext(), value0.getType()));
    }
    auto storeInst = builder.create<inst::Store>(range, value0, pointer);

    if (values.size() > 1) {
      auto value1 = values[1];
      auto const8 =
          createIntConstant(8, IntT::get(module->getContext(), 64, true));
      pointer = builder.create<inst::Gep>(
          range, pointer, const8,
          PointerT::get(module->getContext(), value1.getType()));
      storeInst = builder.create<inst::Store>(range, value1, pointer);
    }
  } else if (totalValueSize > memorySize) {
    assert(totalValueSize / 2 < memorySize);
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

      auto memcpy = createMemCpy(range);
      auto constMemSize = createIntConstant(
          memorySize, IntT::get(module->getContext(), 64, true));
      auto voidPtrT = getVoidPointerT();

      auto voidInt64Ptr =
          builder.create<inst::TypeCast>(range, int64Ptr, voidPtrT);

      if (pointer.getType() != voidPtrT)
        pointer = builder.create<inst::TypeCast>(range, pointer, voidPtrT);

      builder.create<inst::Call>(
          range, memcpy,
          llvm::SmallVector<Value>{pointer, voidInt64Ptr, constMemSize});
    } else {
      auto value0 = values[0];
      auto value1 = values[1];

      auto store0 = builder.create<inst::Store>(range, value0, int64_2Ptr);
      auto const8 =
          createIntConstant(8, IntT::get(module->getContext(), 64, true));
      auto pointer1 = builder.create<inst::Gep>(
          range, int64_2Ptr, const8,
          PointerT::get(module->getContext(), value1.getType()));

      auto store1 = builder.create<inst::Store>(range, value1, pointer1);

      auto memcpy = createMemCpy(range);
      auto constMemSize = createIntConstant(
          memorySize, IntT::get(module->getContext(), 64, true));
      auto voidPtrT = getVoidPointerT();
      auto voidInt64_2Ptr =
          builder.create<inst::TypeCast>(range, int64_2Ptr, voidPtrT);
      if (pointer.getType() != voidPtrT)
        pointer = builder.create<inst::TypeCast>(range, pointer, voidPtrT);
      builder.create<inst::Call>(
          range, memcpy,
          llvm::SmallVector<Value>{pointer, voidInt64_2Ptr, constMemSize});
    }
  } else
    llvm_unreachable("Unexpected total value size for struct manipulation."
                     "Because total value size is always bigger than memory.");
}

} // namespace kecc::ir
