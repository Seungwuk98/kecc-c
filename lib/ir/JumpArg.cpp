#include "kecc/ir/JumpArg.h"
#include "kecc/ir/Block.h"
#include "llvm/Support/MemAlloc.h"

namespace kecc::ir {

JumpArg *JumpArg::create(InstructionStorage *storage, Block *block,
                         llvm::ArrayRef<Value> args) {
  auto sizeToAlloc = totalSizeToAlloc<Operand>(args.size());
  void *rawMem = llvm::safe_malloc(sizeToAlloc);

  auto *jumpArg = new (rawMem) JumpArg(storage, block, args.size());

  Operand *operands = jumpArg->getTrailingObjects<Operand>();

  for (size_t i = 0; i < args.size(); ++i) {
    new (&operands[i]) Operand(storage, args[i]);
  }

  return jumpArg;
}

JumpArg *JumpArg::create(InstructionStorage *storage,
                         const JumpArgState &state) {
  return create(storage, state.block, state.args);
}

Block *JumpArg::getBlock() const { return block; }
llvm::ArrayRef<Operand> JumpArg::getArgs() const {
  return llvm::ArrayRef(getTrailingObjects<Operand>(), operandSize);
}

JumpArgState JumpArg::getAsState() const {
  llvm::SmallVector<Value> args;
  args.reserve(operandSize);
  const Operand *operands = getTrailingObjects<Operand>();
  for (size_t i = 0; i < operandSize; ++i)
    args.emplace_back(operands[i]);
  return {block, args};
}

WalkResult
JumpArg::walk(llvm::function_ref<WalkResult(const Operand &)> callback) {
  Operand *operands = getTrailingObjects<Operand>();

  for (size_t i = 0; i < operandSize; ++i) {
    auto result = callback(operands[i]);
    if (result.isInterrupt())
      return result;
    if (result.isSkip())
      return WalkResult::advance();
  }

  return WalkResult::advance();
}

void JumpArg::replaceOperand(Value oldV, Value newV) {
  Operand *operands = getTrailingObjects<Operand>();

  for (size_t i = 0; i < operandSize; ++i) {
    if (operands[i] == oldV)
      operands[i] = Operand(owner, newV);
  }
}

void JumpArg::destroy() {
  // Explicitly destroy operands to update def-use chain
  for (size_t i = 0; i < operandSize; ++i) {
    getTrailingObjects<Operand>()[i].~Operand();
  }
  // Free the memory allocated for this JumpArg
  free(reinterpret_cast<char *>(this));
}

void JumpArg::printJumpArg(IRPrintContext &context) const {
  context.printJumpArg(this);
}

void JumpArg::dropReferences() {
  Operand *operands = getTrailingObjects<Operand>();
  for (size_t i = 0; i < operandSize; ++i) {
    operands[i].drop();
  }
}

} // namespace kecc::ir
