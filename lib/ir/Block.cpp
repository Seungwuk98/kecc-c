#include "kecc/ir/Block.h"
#include "kecc/ir/Context.h"
#include "kecc/ir/IR.h"
#include "kecc/ir/Instruction.h"

namespace kecc::ir {

WalkResult
Block::walk(llvm::function_ref<WalkResult(InstructionStorage *)> callback) {
  for (auto inst : *this) {
    auto result = callback(inst);
    if (result.isInterrupt())
      return result;
    if (result.isSkip())
      break;
  }
  return WalkResult::advance();
}

Block::Iterator Block::phiEnd() const {
  auto it = phiBegin();
  for (; it != end(); ++it) {
    if (!(*it)->getDefiningInst<Phi>())
      return it;
  }
  return it;
}

Block::Iterator Block::tempEnd() const {
  auto endIt = end();
  endIt--;
  if (*endIt == nullptr)
    return end();
  if ((*endIt)->getDefiningInst<BlockExit>())
    return endIt;
  return end();
}

BlockExit Block::getExit() const {
  auto tail = end();
  tail--;
  auto exit = (*tail)->getDefiningInst<BlockExit>();
  assert(exit && "Last instruction must be a block exit");
  return exit;
}

Block::InsertionPoint Block::getLastInsertionPoint() {
  auto lastIt = end();
  lastIt--;
  return InsertionPoint(this, lastIt);
}

Block::InsertionPoint Block::getStartInsertionPoint() {
  auto firstInst = begin();
  return InsertionPoint(this, --firstInst);
}

Block::InsertionPoint Block::getLastTempInsertionPoint() {
  auto ip = getLastInsertionPoint();
  if ((*ip.getIterator())->getDefiningInst<BlockExit>())
    --ip;
  return ip;
}

void Block::registerRid(IRPrintContext &printContext) const {
  auto E = phiEnd();

  std::size_t phiCount = 0;
  for (auto I = phiBegin(); I != E; ++I) {
    auto phi = (*I)->getDefiningInst<Phi>();
    auto rid = RegisterId::arg(phi.getRange(), getId(), phiCount++);
    printContext.setId(phi, rid);
  }

  std::size_t instCount = 0;
  for (auto I = E, E = end(); I != E; ++I) {
    auto *storage = (*I);
    for (auto idx = 0u; idx < storage->getResultSize(); ++idx) {
      auto rid = RegisterId::temp(storage->getRange(), getId(), instCount++);
      printContext.setId(storage->getResult(idx), rid);
    }
  }
}

void Block::erase() {
  auto func = getParentFunction();
  func->eraseBlock(getId());
}

void Block::print(IRPrintContext &printContext) const {
  printContext.getOS() << "block b" << blockId << ":";
  IRPrintContext::AddIndent addIndent(printContext);

  auto E = phiEnd();

  std::size_t phiCount = 0;
  for (auto I = phiBegin(); I != E; ++I) {
    printContext.printIndent();
    auto phi = (*I)->getDefiningInst<Phi>();
    phi.printAsOperand(printContext, true);
  }

  std::size_t instCount = 0;
  for (auto I = E, E = end(); I != E; ++I) {
    printContext.printIndent();
    (*I)->print(printContext);
  }
}

void Block::dump() const { print(llvm::errs()); }

void Block::print(llvm::raw_ostream &os) const {
  IRPrintContext context(os);
  getParentFunction()->registerAllInstInfo(context);
  print(context);
}

void Block::remove(InstructionStorage *inst) { inst->getBlockNode()->remove(); }

Block::Iterator Block::find(InstructionStorage *inst) const {
  return Block::Iterator(inst->getBlockNode());
}

ir::IR *Block::getParentIR() const {
  return getParentFunction()->getParentIR();
}

void Block::dropReferences() {
  walk([&](InstructionStorage *inst) -> WalkResult {
    inst->dropReferences();
    return WalkResult::advance();
  });
}

} // namespace kecc::ir
