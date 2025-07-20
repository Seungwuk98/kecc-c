#ifndef KECC_IR_JUMP_ARG
#define KECC_IR_JUMP_ARG

#include "kecc/ir/Instruction.h"
#include "kecc/ir/WalkSupport.h"
#include "llvm/ADT/STLExtras.h"

namespace kecc::ir {
class JumpArg final : public llvm::TrailingObjects<JumpArg, Operand> {
public:
  JumpArg(const JumpArg &) = delete;
  JumpArg &operator=(const JumpArg &) = delete;

  static JumpArg *create(InstructionStorage *storage, Block *block,
                         llvm::ArrayRef<Value> args);

  static JumpArg *create(InstructionStorage *storage,
                         const JumpArgState &state);

  Block *getBlock() const;
  llvm::ArrayRef<Operand> getArgs() const;

  void destroy();

  WalkResult walk(llvm::function_ref<WalkResult(const Operand &)> callback);

  void replaceOperand(Value oldV, Value newV);

  void printJumpArg(IRPrintContext &context) const;

  JumpArgState getAsState() const;

  bool isEqual(const JumpArg *other) const {
    if (this == other)
      return true;

    if (operandSize != other->operandSize || block != other->block)
      return false;

    return llvm::equal(getArgs(), other->getArgs());
  }

  bool operator==(const JumpArg &other) const { return isEqual(&other); }
  bool operator!=(const JumpArg &other) const { return !isEqual(&other); }

  void dropReferences();

private:
  JumpArg(InstructionStorage *owner, Block *block, size_t operandSize)
      : block(block), operandSize(operandSize) {}
  InstructionStorage *owner;
  Block *block;
  const size_t operandSize;
};

} // namespace kecc::ir
#endif // KECC_IR_JUMP_ARG
