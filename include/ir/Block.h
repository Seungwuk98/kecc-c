#ifndef KECC_IR_BLOCK_H
#define KECC_IR_BLOCK_H

#include "ir/Instructions.h"
#include "utils/List.h"
#include <cstdint>

namespace kecc::ir {

class Function;

class Block : public utils::ListObject<Instruction *> {
public:
  Block(int id) : blockId(id) {}

  class InsertionPoint {
  public:
    InsertionPoint(Iterator it) : it(it) {}

    InsertionPoint insertNext(Instruction *inst) {
      assert(it.getNode()->next && "Cannot insert after the tail of list");
      it.getNode()->insertNext(inst);
      return InsertionPoint(it.getNode()->next);
    }

    void insertBefore(Instruction *inst) {
      assert(it.getNode()->prev && "Cannot insert before the head of list");
      it.getNode()->prev->insertNext(inst);
    }

  private:
    Iterator it;
  };

  Iterator phiBegin() const { return begin(); }
  Iterator phiEnd() const;

  Iterator instBegin() const { return phiEnd(); }
  Iterator instEnd() const { return end(); }

  void setParentFunction(Function *func) { parentFunction = func; }

private:
  int blockId;
  Function *parentFunction = nullptr;
};

} // namespace kecc::ir

#endif // KECC_IR_BLOCK_H
