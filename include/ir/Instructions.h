#ifndef KECC_IR_INSTRUCTIONS_H
#define KECC_IR_INSTRUCTIONS_H

#include "ir/Operand.h"
#include "ir/Types.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/TrailingObjects.h"

namespace kecc::ir {

class Instruction;
class Block;

class Instruction : public llvm::TrailingObjects<Instruction, Type, Operand> {
public:
  llvm::ArrayRef<Type> getTypes() const;
  llvm::ArrayRef<Operand> getOperands() const;

private:
  llvm::SMRange range;
  Block *parentBlock = nullptr;
};

} // namespace kecc::ir

#endif // KECC_IR_INSTRUCTIONS_H
