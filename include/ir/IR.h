#ifndef KECC_IR_H
#define KECC_IR_H

#include "ir/Block.h"
#include "ir/Types.h"
#include "utils/List.h"
#include "llvm/ADT/StringRef.h"

namespace kecc::ir {

class IR : public utils::ListObject<Function *> {
public:
  IR() {
    structBlock = new Block(-1);
    globalBlock = new Block(-2);
  }

  ~IR() {
    delete structBlock;
    delete globalBlock;
  }

private:
  Block *structBlock;
  Block *globalBlock;
};

class Function : public utils::ListObject<Block *> {
public:
private:
  llvm::StringRef name;
  Type functionType;
  Block *entryBlock = nullptr;
  llvm::SmallVector<Instruction *> allocations;
  IR *parentProgram = nullptr;
};

} // namespace kecc::ir

#endif // KECC_IR_H
