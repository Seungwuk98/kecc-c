#include "kecc/ir/Module.h"
#include "kecc/ir/IRAttributes.h"
#include "kecc/ir/IRBuilder.h"
#include "kecc/ir/Instruction.h"
#include "kecc/ir/JumpArg.h"
#include "kecc/ir/WalkSupport.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVectorExtras.h"

namespace kecc::ir {

const llvm::DenseSet<Block *> &Module::getSuccessors(Block *block) const {
  return successors.at(block);
}

const llvm::DenseSet<Block *> &Module::getPredecessors(Block *block) const {
  return predecessors.at(block);
}

std::unique_ptr<Module> Module::create(std::unique_ptr<IR> ir) {
  auto module = std::unique_ptr<Module>(new Module(std::move(ir)));

  module->updatePredsAndSuccs();
  return module;
}

void Module::updatePredsAndSuccs() {
  predecessors.clear();
  successors.clear();
  // Initialize users, predecessors, and successors
  for (const auto function : *ir) {
    for (const auto block : *function) {
      predecessors.try_emplace(block, llvm::DenseSet<Block *>());
      successors.try_emplace(block, llvm::DenseSet<Block *>());

      auto exit = block->getExit();
      exit.walk([&](JumpArg *arg) -> WalkResult {
        auto succ = arg->getBlock();
        successors[block].insert(succ);
        predecessors[succ].insert(block);
        return WalkResult::advance();
      });
    }
  }
}

void Module::replaceInst(InstructionStorage *oldInst,
                         InstructionStorage *newInst) {
  if (oldInst == newInst) {
    return; // No need to replace if they are the same
  }

  assert(!oldInst->getDefiningInst<BlockExit>() &&
         !newInst->getDefiningInst<BlockExit>() &&
         "Cannot replace a block exit instruction by this function");

  auto oldBlock = oldInst->getParentBlock();
  auto newBlock = newInst->getParentBlock();

  replaceInst(oldInst, newInst->getResults());

  auto it = oldBlock->find(oldInst);
  assert(it != oldBlock->end() && "Old instruction must be in the block");

  it.getNode()->data = newInst;

  it = newBlock->find(newInst);

  it.getNode()->data = nullptr;
  it.getNode()->remove();
  newInst->setParentBlock(oldBlock);
}

bool Module::replaceInst(
    InstructionStorage *oldInst,
    llvm::function_ref<InstructionStorage *(IRBuilder &, InstructionStorage *)>
        newInstBuildFunc,
    bool remove) {
  assert(oldInst && "Cannot replace a null InstructionStorage");
  assert(!oldInst->getDefiningInst<BlockExit>() &&
         "Cannot replace a block exit instruction by this function");

  IRBuilder builder(oldInst->getContext());
  auto it = oldInst->getParentBlock()->find(oldInst);
  assert(it != oldInst->getParentBlock()->end() &&
         "Old instruction must be in the block");

  builder.setInsertionPoint(
      Block::InsertionPoint(oldInst->getParentBlock(), it));

  auto newInst = newInstBuildFunc(builder, oldInst);
  if (!newInst)
    return false;

  assert(!oldInst->getContext()->diag().hasError() &&
         "Error in new instruction creation, cannot replace instruction");

  if (llvm::all_of(oldInst->getResults(),
                   [](Value V) { return !V.hasUses(); })) {
    if (remove)
      it.getNode()->remove();
  } else {
    replaceInst(oldInst, newInst->getResults(), remove);
  }
  return true;
}

void Module::replaceInst(InstructionStorage *oldInst,
                         llvm::ArrayRef<Value> newVals, bool remove) {
  assert(oldInst->getResultSize() == newVals.size() &&
         "New values must match the size of the old instruction results");

  for (std::size_t i = 0; i < oldInst->getResultSize(); ++i) {
    auto oldV = oldInst->getResult(i);
    auto newV = newVals[i];
    assert(oldV.getType() == newV.getType() &&
           "Cannot replace value with different type");
    oldV.replaceWith(newV);
  }

  if (remove) {
    Block *block = oldInst->getParentBlock();
    block->remove(oldInst);
  }
}

bool Module::replaceExit(
    BlockExit oldExit,
    llvm::function_ref<BlockExit(IRBuilder &builder, BlockExit oldExit)>
        newExitBuildFunc) {

  assert(oldExit && "Cannot replace a null BlockExit instruction");

  Block *block = oldExit.getParentBlock();
  auto lastIter = block->end();
  lastIter--;

  assert((*lastIter == oldExit.getStorage()) &&
         "Last instruction in the block must be a BlockExit instruction");

  IRBuilder builder(oldExit.getContext());
  builder.setInsertionPoint(block);

  auto newExit = newExitBuildFunc(builder, oldExit);
  if (!newExit)
    // If new block exit is null, it means that this block exit is not needed to
    // replace
    return false;

  assert(!oldExit.getContext()->diag().hasError() &&
         "Error in new exit creation, cannot replace exit instruction");

  auto oldSuccessors = getSuccessors(block);
  // init successors
  successors[block].clear();
  for (auto oldSucc : oldSuccessors) {
    // Remove predecessors from successors
    predecessors[oldSucc].erase(block);
  }

  newExit.walk([&](JumpArg *arg) -> WalkResult {
    auto succ = arg->getBlock();
    successors[block].insert(succ);
    predecessors[succ].insert(block);
    return WalkResult::advance();
  });

  if (oldExit != newExit) {
    // remove old exit
    lastIter.getNode()->remove();
  }

  return true; // replaced
}

void Module::removeInst(InstructionStorage *inst) {
#ifndef NDEBUG
  for (auto idx = 0u; idx < inst->getResultSize(); ++idx) {
    auto value = inst->getResult(idx);
    assert(value.useBegin() == value.useEnd() &&
           "Instruction is still used by other instructions after removal");
  }
#endif // NDEBUG

  Block *block = inst->getParentBlock();
  block->remove(inst);
}

void Module::removeBlock(Block *block) {
#ifndef NDEBUG
  for (auto inst : *block) {
    for (auto idx = 0u; idx < inst->getResultSize(); ++idx) {
      auto value = inst->getResult(idx);
      for (auto I = value.useBegin(), E = value.useEnd(); I != E; ++I) {
        Operand *user = *I;
        Instruction(user->getOwner()).dump();
        assert(user->getOwner()->getParentBlock() != block &&
               "Instruction is still used by other instructions after block "
               "removal");
      }
    }
  }
#endif // NDEBUG
  if (auto it = predecessors.find(block); it != predecessors.end()) {
    for (auto pred : it->second) {
      successors[pred].erase(block);
    }
    predecessors.erase(it);
  }
  if (auto it = successors.find(block); it != successors.end()) {
    for (auto succ : it->second) {
      predecessors[succ].erase(block);
    }
    successors.erase(it);
  }
  block->erase();
}

IRContext *Module::getContext() const { return getIR()->getContext(); }

} // namespace kecc::ir
