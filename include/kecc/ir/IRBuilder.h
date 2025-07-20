#ifndef KECC_IR_BUILDER_H
#define KECC_IR_BUILDER_H

#include "kecc/ir/Block.h"
#include "kecc/ir/Context.h"
#include "kecc/ir/Instruction.h"
#include "llvm/Support/SMLoc.h"

namespace kecc::ir {

template <typename Inst>
concept HasInstVerify = requires(Inst inst) {
  { inst.verify() } -> std::convertible_to<bool>;
};

class IRBuilder {
public:
  IRBuilder(IRContext *context) : context(context) {}

  IRContext *getContext() const { return context; }

  template <typename Inst, typename... Args>
  Inst create(llvm::SMRange range, Args &&...args) {
    auto typeId = TypeID::get<Inst>();

    InstructionState state;
    state.setRange(range);
    state.setParentBlock(insertionPoint.getBlock());

    Inst::build(*this, state, std::forward<Args>(args)...);
    auto *storage = InstructionStorage::create(state);
    storage->setAbstractInstruction(getAbstractInstruction(typeId));

    Inst newInst(storage);
    insertionPoint = insertionPoint.insertNext(newInst.getStorage());
    return newInst;
  }

  // create clone and insert to insertion point
  InstructionStorage *clone(InstructionStorage *target);

  AbstractInstruction *getAbstractInstruction(TypeID typeId) const {
    return context->getTypeStorage()->getAbstractTable<AbstractInstruction>(
        typeId);
  }

  struct InsertionGuard {
    InsertionGuard(IRBuilder &builder, Block::InsertionPoint point)
        : builder(builder), savedInsertionPoint(builder.insertionPoint) {
      builder.insertionPoint = point;
    }

    InsertionGuard(IRBuilder &builder, Block *block)
        : builder(builder), savedInsertionPoint(builder.insertionPoint) {
      builder.insertionPoint = block->getLastInsertionPoint();
    }

    ~InsertionGuard() { builder.insertionPoint = savedInsertionPoint; }

  private:
    IRBuilder &builder;
    Block::InsertionPoint savedInsertionPoint;
  };

  void setInsertionPoint(Block *block) {
    insertionPoint = block->getLastInsertionPoint();
  }

  void setInsertionPoint(Block::InsertionPoint point) {
    insertionPoint = point;
  }

private:
  Block::InsertionPoint insertionPoint;
  IRContext *context;
};

} // namespace kecc::ir

#endif // KECC_IR_BUILDER_H
