#ifndef KECC_ASM_BUILDER_H
#define KECC_ASM_BUILDER_H

#include "kecc/asm/Asm.h"
#include "kecc/asm/AsmInstruction.h"

namespace kecc::as {

class AsmBuilder {
public:
  AsmBuilder() {}

  struct InsertionGuard {
    InsertionGuard(AsmBuilder &builder, Block::InsertionPoint point)
        : builder(builder), previousPoint(builder.insertionPoint) {
      builder.setInsertionPoint(point);
    }

    InsertionGuard(AsmBuilder &builder)
        : builder(builder), previousPoint(builder.insertionPoint) {}

    ~InsertionGuard() { builder.setInsertionPoint(previousPoint); }

  private:
    AsmBuilder &builder;
    Block::InsertionPoint previousPoint;
  };

  void setInsertionPoint(Block::InsertionPoint point) {
    insertionPoint = point;
  }

  void setInsertionPointLast(Block *block) {
    assert(block && "Cannot set insertion point to a null block");
    auto endNode = block->getTail();
    auto lastNode = endNode->prev;
    auto insertionPoint = Block::InsertionPoint(block, lastNode);
    setInsertionPoint(insertionPoint);
  }

  void setInsertionPointStart(Block *block) {
    assert(block && "Cannot set insertion point to a null block");
    auto startNode = block->getHead();
    auto insertionPoint = Block::InsertionPoint(block, startNode);
    setInsertionPoint(insertionPoint);
  }

  Block::InsertionPoint getInsertionPoint() const { return insertionPoint; }

  void setInsertionPointAfterInst(Instruction *inst) {
    auto node = inst->getNode();
    auto insertionPoint = Block::InsertionPoint(inst->getParentBlock(), node);
    setInsertionPoint(insertionPoint);
  }

  void setInsertionPointBeforeInst(Instruction *inst) {
    setInsertionPointAfterInst(inst);
    insertionPoint--;
  }

  template <typename Inst, typename... Args> Inst *create(Args &&...args) {
    assert(insertionPoint.isValid() &&
           "Insertion point is not valid, cannot insert instruction");
    auto *inst = new Inst(std::forward<Args>(args)...);
    auto newPoint = insertionPoint.insertNext(inst);
    inst->setParent(insertionPoint.getBlock());
    inst->setNode(newPoint.getIterator().getNode());
    setInsertionPoint(newPoint);
    return inst;
  }

private:
  Block::InsertionPoint insertionPoint;
};

} // namespace kecc::as

#endif // KECC_ASM_BUILDER_H
