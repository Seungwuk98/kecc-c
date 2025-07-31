#ifndef KECC_IR_BLOCK_H
#define KECC_IR_BLOCK_H

#include "kecc/ir/Instruction.h"
#include "kecc/ir/WalkSupport.h"
#include "kecc/utils/List.h"

namespace kecc::ir {

class Function;
class IR;

class Block : public utils::ListObject<Block, InstructionStorage *> {
public:
  Block(int id) : blockId(id) {}

  static void deleteObject(InstructionStorage *storage) {
    if (storage) {
      storage->destroy();
    }
  }

  class InsertionPoint {
  public:
    InsertionPoint() : block(nullptr), it(nullptr) {}
    InsertionPoint(Block *block, Iterator it) : block(block), it(it) {}

    InsertionPoint insertNext(InstructionStorage *inst) {
      assert(it.getNode()->next && "Cannot insert after the tail of list");
      Node *newNode = it.getNode()->insertNext(inst);
      return InsertionPoint(block, Iterator(newNode));
    }

    void insertBefore(InstructionStorage *inst) {
      assert(it.getNode()->prev && "Cannot insert before the head of list");
      it.getNode()->prev->insertNext(inst);
    }

    Block *getBlock() const { return block; }
    Iterator getIterator() const { return it; }

    bool isValid() const { return block != nullptr && it.getNode() != nullptr; }

    InsertionPoint &operator++() {
      it++;
      return *this;
    }

    InsertionPoint operator++(int) {
      InsertionPoint temp = *this;
      ++(*this);
      return temp;
    }

    InsertionPoint &operator--() {
      it--;
      return *this;
    }

    InsertionPoint operator--(int) {
      InsertionPoint temp = *this;
      --(*this);
      return temp;
    }

  private:
    Block *block;
    Iterator it;
  };

  Iterator phiBegin() const { return begin(); }
  Iterator phiEnd() const;

  Iterator tempBegin() const { return phiEnd(); }
  Iterator tempEnd() const;
  Iterator instEnd() const { return end(); }

  BlockExit getExit() const;

  Iterator find(InstructionStorage *inst) const;

  void erase();

  void setParentFunction(Function *func) { parentFunction = func; }

  InsertionPoint getLastInsertionPoint();
  InsertionPoint getStartInsertionPoint();
  InsertionPoint getLastTempInsertionPoint();

  int getId() const { return blockId; }

  IR *getParentIR() const;
  Function *getParentFunction() const { return parentFunction; }

  WalkResult
  walk(llvm::function_ref<WalkResult(InstructionStorage *)> callback);

  void registerRid(IRPrintContext &printContext) const;
  void print(IRPrintContext &printContext) const;
  void dump() const;
  void print(llvm::raw_ostream &os) const;
  void remove(InstructionStorage *inst);

  void dropReferences();

private:
  int blockId;
  Function *parentFunction = nullptr;
};

} // namespace kecc::ir

#endif // KECC_IR_BLOCK_H
