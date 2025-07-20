#ifndef KECC_IR_H
#define KECC_IR_H

#include "kecc/ir/Block.h"
#include "kecc/ir/Context.h"
#include "kecc/ir/Type.h"
#include "kecc/ir/WalkSupport.h"
#include "kecc/utils/List.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

namespace kecc::ir {

class IR : public utils::ListObject<IR, Function *> {
public:
  IR(IRContext *context) : context(context) {
    structBlock = new Block(-1);
    globalBlock = new Block(-2);
    constantBlock = new Block(-3);
  }

  ~IR();

  Block *getStructBlock() const { return structBlock; }
  Block *getGlobalBlock() const { return globalBlock; }
  Block *getConstantBlock() const { return constantBlock; }

  void addFunction(Function *function);
  Function *getFunction(llvm::StringRef name) const;
  void erase(llvm::StringRef name);
  std::size_t getFunctionCount() const { return functionMap.size(); }

  void print(IRPrintContext &printContext) const;

  IRContext *getContext() const { return context; }

private:
  Block *structBlock;
  Block *globalBlock;
  Block *constantBlock;
  llvm::StringMap<Node *> functionMap;
  IRContext *context;
};

class Function : public utils::ListObject<Function, Block *> {
public:
  Function(llvm::StringRef name, Type functionType, IR *parentProgram,
           IRContext *context);
  ~Function();

  IR *getParentIR() const { return parentProgram; }

  // Adds a new block with the given ID to the function.
  // If a block with the same ID already exists, it returns the existing block
  // and pushes it to the end of the block list.
  Block *addBlock(int id);
  Block *getAllocationBlock() const;
  Block *getUnresolvedBlock() const;
  Block *getEntryBlock() const;

  bool blockExists(int id) const { return blockMap.contains(id); }
  void setEntryBlock(int bid) { entryBid = bid; }

  Block *getBlockById(int id) const {
    auto it = blockMap.find(id);
    return it != blockMap.end() ? it->second->data : nullptr;
  }

  WalkResult
  walk(llvm::function_ref<WalkResult(InstructionStorage *)> callback);

  WalkResult walk(llvm::function_ref<WalkResult(const Operand &)> callback);

  llvm::StringRef getName() const { return name; }
  Type getFunctionType() const { return functionType; }
  std::size_t getBlockCount() const { return blockMap.size(); }

  void print(IRPrintContext &context) const;

  void eraseBlock(int id);

  IRContext *getContext() const { return context; }

  void registerAllInstInfo(IRPrintContext &printContext) const;

  void dropReferences();

private:
  std::string name;
  Type functionType;
  IR *parentProgram;
  int entryBid;
  llvm::DenseMap<int, Node *> blockMap;
  Block *allocationBlock;
  Block *unresolvedBlock;
  IRContext *context;
};

} // namespace kecc::ir

#endif // KECC_IR_H
