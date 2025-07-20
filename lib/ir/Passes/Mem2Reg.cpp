#include "kecc/ir/IRAnalyses.h"
#include "kecc/ir/IRAttributes.h"
#include "kecc/ir/IRInstructions.h"
#include "kecc/ir/IRTransforms.h"
#include "kecc/ir/Instruction.h"
#include "kecc/ir/WalkSupport.h"
#include "llvm/ADT/DenseSet.h"

namespace kecc::ir {

class JoinTable {
public:
  JoinTable(const DominatorTree *domTree,
            const llvm::DenseMap<std::pair<inst::LocalVariable, Block *>, Value>
                *endValues)
      : domTree(domTree), endValues(endValues) {}

  Block *lookup(inst::LocalVariable localVar, Block *block);
  void buildJoinFromStores(
      const llvm::DenseMap<inst::LocalVariable, llvm::SmallVector<Block *>>
          &stores,
      const llvm::DenseSet<inst::LocalVariable> &inpromotable);

private:
  llvm::DenseMap<inst::LocalVariable, llvm::DenseSet<Block *>> joins;
  const llvm::DenseMap<std::pair<inst::LocalVariable, Block *>, Value>
      *endValues;
  const DominatorTree *domTree;
};

void JoinTable::buildJoinFromStores(
    const llvm::DenseMap<inst::LocalVariable, llvm::SmallVector<Block *>>
        &stores,
    const llvm::DenseSet<inst::LocalVariable> &inpromotable) {
  for (auto [localVar, blocks] : stores) {
    if (inpromotable.contains(localVar))
      continue;

    llvm::DenseSet<Block *> visited;
    while (!blocks.empty()) {
      Block *block = blocks.back();
      blocks.pop_back();

      for (Block *frontier : domTree->getDF(block))
        if (visited.insert(frontier).second)
          blocks.emplace_back(frontier);
    }

    joins[localVar] = std::move(visited);
  }
}

Block *JoinTable::lookup(inst::LocalVariable localVar, Block *block) {
  while (true) {
    if (endValues->contains({localVar, block}))
      break;

    if (joins[localVar].contains(block))
      break;

    if (auto idom = domTree->getIdom(block))
      block = idom;
    else
      break;
  }
  return block;
}

class Mem2RegImpl {
public:
  Mem2RegImpl(Module *module, const DominatorTree *domTree,
              VisitOrderAnalysis *visitOrderAnalysis)
      : module(module), domTree(domTree),
        visitOrderAnalysis(visitOrderAnalysis), joinTable(domTree, &endValues) {
  }
  PassResult run(Function *function);

  void markInpromotable(Value value) {
    auto localVar = value.getDefiningInst<inst::LocalVariable>();
    if (localVar)
      inpromotable.insert(localVar);
  }

private:
  Value createUnresolved(Function *func, inst::LocalVariable localVar);
  std::pair<Value, bool> getOrCreateEndValue(inst::LocalVariable localVar,
                                             Block *block);

  Phi insertPhi(Block *block, inst::LocalVariable localVar);

  Module *module;
  const DominatorTree *domTree;
  VisitOrderAnalysis *visitOrderAnalysis;
  llvm::DenseSet<inst::LocalVariable> inpromotable;
  llvm::DenseMap<inst::LocalVariable, llvm::SmallVector<Block *>> stores;
  llvm::DenseMap<std::pair<inst::LocalVariable, Block *>, Value> endValues;
  JoinTable joinTable;
};

void Mem2Reg::init(Module *module, Function *function) {
  DominanceAnalysis *domAnalysis = module->getAnalysis<DominanceAnalysis>();
  if (!domAnalysis) {
    auto createdDomAnalysis = DominanceAnalysis::create(module);
    assert(createdDomAnalysis && "Failed to create DominanceAnalysis");
    module->insertAnalysis<DominanceAnalysis>(std::move(createdDomAnalysis));
    domAnalysis = module->getAnalysis<DominanceAnalysis>();
  }

  VisitOrderAnalysis *visitOrderAnalysis =
      module->getAnalysis<VisitOrderAnalysis>();
  if (!visitOrderAnalysis) {
    auto createdVisitOrderAnalysis = VisitOrderAnalysis::create(module);
    assert(createdVisitOrderAnalysis && "Failed to create VisitOrderAnalysis");
    module->insertAnalysis<VisitOrderAnalysis>(
        std::move(createdVisitOrderAnalysis));
    visitOrderAnalysis = module->getAnalysis<VisitOrderAnalysis>();
  }

  impl = std::make_unique<Mem2RegImpl>(
      module, domAnalysis->getDominatorTree(function), visitOrderAnalysis);
}

void Mem2Reg::exit(Module *module, Function *function) { impl.reset(); }

PassResult Mem2Reg::run(Module *module, Function *function) {
  return impl->run(function);
}
Mem2Reg::Mem2Reg() = default;
Mem2Reg::~Mem2Reg() = default;

Value Mem2RegImpl::createUnresolved(Function *func,
                                    inst::LocalVariable localVar) {
  IRBuilder builder(module->getContext());
  builder.setInsertionPoint(func->getUnresolvedBlock());

  auto type = localVar.getType().cast<PointerT>().getPointeeType();
  auto unresolved = builder.create<inst::Unresolved>(localVar.getRange(), type);

  return unresolved;
}

Phi Mem2RegImpl::insertPhi(Block *block, inst::LocalVariable localVar) {
  IRBuilder builder(module->getContext());
  auto it = block->phiEnd();
  builder.setInsertionPoint(Block::InsertionPoint(block, --it));

  auto phi =
      builder.create<Phi>(localVar.getRange(),
                          localVar.getType().cast<PointerT>().getPointeeType());
  phi.setValueName(localVar.getValueName());

  auto preds = module->getPredecessors(block);
  for (Block *pred : preds) {
    auto predJoin = joinTable.lookup(localVar, pred);
    auto endValue = endValues.at(
        {localVar, predJoin}); // pred block's end value must be created in pass
    assert(endValue.getType() == phi.getType() && "End value type mismatch");
    module->replaceExit(
        pred->getExit(),
        [&](IRBuilder &builder, BlockExit oldExit) -> BlockExit {
          auto newExit = builder.clone(oldExit.getStorage());

          for (auto [idx, jumpArg] : llvm::enumerate(newExit->getJumpArgs())) {
            if (jumpArg->getBlock() == block) {
              auto state = jumpArg->getAsState();
              state.pushArg(endValue);
              newExit->setJumpArg(idx, state);
            }
          }

          return newExit;
        });
  }

  return phi;
}

std::pair<Value, bool>
Mem2RegImpl::getOrCreateEndValue(inst::LocalVariable localVar, Block *block) {
  auto it = endValues.find({localVar, block});
  if (it != endValues.end()) {
    return {it->second, false};
  }
  // If the end value is not found, create an unresolved value
  // and store it in endValues.
  Value unresolved = createUnresolved(block->getParentFunction(), localVar);
  endValues[{localVar, block}] = unresolved;
  return {unresolved, true};
}

PassResult Mem2RegImpl::run(Function *function) {
  function->walk([&](InstructionStorage *inst) -> WalkResult {
    if (auto storeInst = inst->getDefiningInst<inst::Store>()) {
      auto ptr = storeInst.getPointer();
      auto value = storeInst.getValue();

      if (auto localVar = ptr.getDefiningInst<inst::LocalVariable>()) {
        stores[localVar].emplace_back(storeInst.getParentBlock());
      }

      markInpromotable(value);
    } else if (!inst->getDefiningInst<inst::Load>()) {
      inst->walk([&](const Operand &op) -> WalkResult {
        markInpromotable(op);
        return WalkResult::advance();
      });
    }

    return WalkResult::advance();
  });

  if (utils::all_of(
          *function->getAllocationBlock(), [&](InstructionStorage *localVar) {
            auto localVarInst =
                localVar->getDefiningInst<inst::LocalVariable>();
            assert(localVarInst && "Expected a local variable instruction");
            return inpromotable.contains(localVarInst);
          })) {
    return PassResult::skip();
  }

  joinTable.buildJoinFromStores(stores, inpromotable);
  llvm::DenseMap<Value, Value> replaces;
  llvm::DenseMap<inst::LocalVariable, std::size_t> localVarCounts;
  llvm::DenseSet<std::pair<Block *, inst::LocalVariable>> phiNodes;
  llvm::DenseMap<std::pair<Block *, inst::LocalVariable>, Value> unresolveds;

  IRBuilder builder(module->getContext());
  builder.setInsertionPoint(module->getIR()->getConstantBlock());

  size_t localVarIndex = 0u;
  for (auto localVar : *function->getAllocationBlock()) {
    auto localVarInst = localVar->getDefiningInst<inst::LocalVariable>();
    auto initValue = builder.create<inst::Constant>(
        localVar->getRange(),
        ConstantUndefAttr::get(
            module->getContext(),
            localVarInst.getType().cast<PointerT>().getPointeeType()));
    endValues[{localVarInst, function->getEntryBlock()}] = initValue;
    localVarCounts[localVarInst] = localVarIndex++;
  }

  for (Block *block : visitOrderAnalysis->getReversePostOrder(function)) {
    block->walk([&](InstructionStorage *inst) -> WalkResult {
      if (inst::Store store = inst->getDefiningInst<inst::Store>()) {
        auto ptr = store.getPointer();
        if (auto localVar = ptr.getDefiningInst<inst::LocalVariable>()) {
          if (inpromotable.contains(localVar))
            return WalkResult::advance();

          endValues[{localVar, block}] = store.getValue();
        }
      } else if (inst::Load load = inst->getDefiningInst<inst::Load>()) {
        auto ptr = load.getPointer();
        if (auto localVar = ptr.getDefiningInst<inst::LocalVariable>()) {
          if (inpromotable.contains(localVar))
            return WalkResult::advance();

          Block *joinBlock = joinTable.lookup(localVar, block);
          auto [endValue, created] = getOrCreateEndValue(localVar, joinBlock);
          if (created) {
            phiNodes.insert({joinBlock, localVar});
            unresolveds.try_emplace({joinBlock, localVar}, endValue);
          }

          endValues.try_emplace({localVar, block}, endValue);
          auto [_, inserted] = replaces.try_emplace(load, endValue);
          assert(inserted && "Load should not be replaced multiple times");
          (void)inserted;
        }
      }
      return WalkResult::advance();
    });
  }

  llvm::DenseSet<std::pair<Block *, inst::LocalVariable>> phiNodeVisited =
      phiNodes;
  llvm::SmallVector<std::pair<Block *, inst::LocalVariable>> phiNodeStack(
      phiNodes.begin(), phiNodes.end());

  while (!phiNodeStack.empty()) {
    auto [block, localVar] = phiNodeStack.back();
    phiNodeStack.pop_back();

    auto preds = module->getPredecessors(block);
    for (Block *pred : preds) {
      Block *predJoin = joinTable.lookup(localVar, pred);
      auto [endValue, created] = getOrCreateEndValue(localVar, predJoin);
      if (created) {
        if (phiNodeVisited.insert({predJoin, localVar}).second)
          phiNodeStack.emplace_back(predJoin, localVar);

        unresolveds.try_emplace({predJoin, localVar}, endValue);
      }
      endValues.try_emplace({localVar, block}, endValue);
    }
  }

  llvm::SmallVector<std::pair<Block *, inst::LocalVariable>> phiNodeVisitedVec(
      phiNodeVisited.begin(), phiNodeVisited.end());
  std::sort(phiNodeVisitedVec.begin(), phiNodeVisitedVec.end(),
            [&](const auto &l, const auto &r) -> bool {
              return localVarCounts[l.second] < localVarCounts[r.second];
            });

  llvm::DenseMap<Value, Value> phiReplaces;
  for (auto [block, localVar] : phiNodeVisitedVec) {
    auto phi = insertPhi(block, localVar);
    auto unresolved = unresolveds.at({block, localVar});
    auto [_, inserted] = phiReplaces.try_emplace(unresolved, phi);
    assert(inserted &&
           "Unresolved value should not be replaced multiple times");
    (void)inserted;
  }

  // phi insertion can insert unreplaced unresolveds,
  // so we need to replace them after all phis are inserted
  for (auto [unresolved, phi] : phiReplaces) {
    unresolved.replaceWith(phi);
  }

  for (auto [oldValue, newValue] : replaces) {
    while (replaces.contains(newValue)) {
      newValue = replaces[newValue];
    }
    // in this replace `newValue` can be unresolved, so we need to check
    if (auto it = phiReplaces.find(newValue); it != phiReplaces.end()) {
      newValue = it->second;
    }
    oldValue.replaceWith(newValue);
  }

  // Remove all unresolve instructions
  for (auto [unresolved, _] : phiReplaces) {
    module->removeInst(unresolved.getInstruction());
  }

  llvm::SmallVector<InstructionStorage *> promoted;
  function->walk([&](InstructionStorage *inst) -> WalkResult {
    Value ptr = nullptr;
    if (auto store = inst->getDefiningInst<inst::Store>())
      ptr = store.getPointer();
    else if (auto load = inst->getDefiningInst<inst::Load>())
      ptr = load.getPointer();

    if (ptr) {
      if (auto localVar = ptr.getDefiningInst<inst::LocalVariable>())
        if (!inpromotable.contains(localVar))
          promoted.emplace_back(inst);
    }
    return WalkResult::advance();
  });

  for (InstructionStorage *inst : promoted) {
    module->replaceInst(
        inst,
        [&](IRBuilder &builder,
            InstructionStorage *oldInst) -> InstructionStorage * {
          return builder.create<inst::Nop>(oldInst->getRange()).getStorage();
        },
        true);
  }

  return PassResult::success();
}

} // namespace kecc::ir
