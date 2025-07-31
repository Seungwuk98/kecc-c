#ifndef KECC_ASM_BUILDER_H
#define KECC_ASM_BUILDER_H

#include "kecc/asm/Asm.h"
#include "kecc/asm/AsmInstruction.h"

namespace kecc::as {

class AsmBuilder {
public:
  AsmBuilder() {}

  void setInsertionPoint(Block::InsertionPoint point) {
    insertionPoint = point;
  }
  Block::InsertionPoint getInsertionPoint() const { return insertionPoint; }

  void setInsertionPointAfterInst(Instruction *inst) {
    auto node = inst->getNode();
    auto insertionPoint = Block::InsertionPoint(inst->getParentBlock(), node);
    this->insertionPoint = insertionPoint;
  }

  void setInsertionPointBeforeInst(Instruction *inst) {
    setInsertionPointAfterInst(inst);
    insertionPoint--;
  }

  template <typename Inst, typename... Args> Inst *createInst(Args &&...args) {
    assert(insertionPoint.isValid() &&
           "Insertion point is not valid, cannot insert instruction");
    auto *inst = new Inst(std::forward<Args>(args)...);
    auto newPoint = insertionPoint.insertNext(inst);
    inst->setParent(insertionPoint.getBlock());
    inst->setNode(newPoint.getIterator().getNode());
    setInsertionPoint(newPoint);
  }

private:
  Block::InsertionPoint insertionPoint;
};

} // namespace kecc::as

#endif // KECC_ASM_BUILDER_H
