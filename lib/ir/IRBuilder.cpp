#include "kecc/ir/IRBuilder.h"
#include "kecc/ir/Instruction.h"
#include "kecc/ir/JumpArg.h"
#include "llvm/ADT/SmallVectorExtras.h"

namespace kecc::ir {

InstructionStorage *IRBuilder::clone(InstructionStorage *target) {
  auto typeId = target->getAbstractInstruction()->getId();
  InstructionState state;
  state.setRange(target->getRange());
  state.setParentBlock(insertionPoint.getBlock());

  llvm::SmallVector<Value> values = target->getResults();
  llvm::SmallVector<Type> types =
      llvm::map_to_vector(values, [](Value value) { return value.getType(); });
  llvm::SmallVector<Value> operands = llvm::map_to_vector(
      target->getOperands(), [](const Operand &op) -> Value { return op; });
  llvm::SmallVector<Attribute> attrs(target->getAttributes());
  llvm::SmallVector<JumpArgState> jumpArgs =
      llvm::map_to_vector(target->getJumpArgs(),
                          [](const JumpArg *arg) { return arg->getAsState(); });

  state.setOperands(operands);
  state.setTypes(types);
  state.setAttributes(attrs);
  state.setJumpArgs(jumpArgs);

  auto *newStorage = InstructionStorage::create(state);
  newStorage->setAbstractInstruction(target->getAbstractInstruction());

  insertionPoint = insertionPoint.insertNext(newStorage);
  return newStorage;
}

} // namespace kecc::ir
