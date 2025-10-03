#include "kecc/ir/IR.h"
#include "kecc/ir/IRInstructions.h"
#include "kecc/ir/IRTransforms.h"
#include "kecc/ir/Instruction.h"
#include "kecc/ir/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/ErrorHandling.h"
#include <ranges>

namespace kecc::ir {

PassResult SimplifyCFG::run(Module *module) {
  Pass *cfgCP = getPassByName("cfg-constant-prop");
  Pass *cfgEmpty = getPassByName("cfg-empty");
  Pass *cfgMerge = getPassByName("cfg-merge");
  Pass *cfgReach = getPassByName("cfg-reach");
  // Iter(cp -> reach -> merge -> empty ->reach)

  while (true) {
    PassResult result = PassResult::skip();

    for (Pass *pass : {cfgCP, cfgReach, cfgMerge, cfgEmpty, cfgReach}) {
      auto thisResult = pass->run(module);
      if (thisResult.isFailure())
        return thisResult;
      if (thisResult.isSuccess())
        result = thisResult;
    }

    if (result.isSkip())
      break; // No more changes, exit the loop
  }

  return PassResult::success();
}

class CFGConstantPropagation : public FuncPass {
public:
  PassResult run(Module *module, Function *fun) override;

  llvm::StringRef getPassArgument() const override {
    return "cfg-constant-prop";
  }

private:
};

class CFGEmpty : public FuncPass {
public:
  PassResult run(Module *module, Function *fun) override;

  llvm::StringRef getPassArgument() const override { return "cfg-empty"; }

  bool candidateCondition(Module *module, Block *block) const;

  // Before this transformation:
  // block x:
  //   ...
  //   j y
  //
  // block y:
  //   ret v
  //
  // After this transformation:
  // block x:
  //   ...
  //   ret v
  //
  // block y:
  //   ret v
  void connectReturn(Module *module, Block *pred, Block *succ) const;

  // Before this transformation:
  // block x:
  //   ...
  //   exit ..., y(), ...
  //
  // block y:
  //   j z(v)
  //
  // After this transformation:
  // block x:
  //  ...
  //  exit ..., z(v), ...
  //
  // block y:
  //   j z(v)
  void connectJump(Module *module, Block *pred, Block *succ) const;
};

class CFGMerge : public FuncPass {
public:
  void init(Module *, Function *) override { parent.clear(); }
  PassResult run(Module *module, Function *fun) override;

  llvm::StringRef getPassArgument() const override { return "cfg-merge"; }

private:
  Block *find(Block *block);

  void setPar(Block *par, Block *child);

  void merge(Module *module, Block *child, Block *parent) const;

  llvm::DenseMap<Block *, Block *> parent;
};

PassResult CFGConstantPropagation::run(Module *module, Function *fun) {
  if (!fun->hasDefinition())
    return PassResult::skip();

  bool changed = false;
  for (Block *block : *fun) {
    changed |= module->replaceExit(
        block->getExit(),
        [&](IRBuilder &builder, BlockExit oldExit) -> BlockExit {
          if (auto br = oldExit.dyn_cast<inst::Branch>()) {
            auto ifArg = br.getIfArg();
            auto elseArg = br.getElseArg();
            if (*ifArg == *elseArg) {
              auto jump = builder.create<inst::Jump>(oldExit.getRange(),
                                                     ifArg->getAsState());
              return jump;
            }

            auto cond = br.getCondition();
            auto constantInst =
                cond.getInstruction()->getDefiningInst<inst::Constant>();
            if (!constantInst)
              return nullptr;

            auto constantVal = constantInst.getValue();
            auto constantIntVal = constantVal.dyn_cast<ConstantIntAttr>();
            if (!constantIntVal)
              return nullptr;
            assert(constantIntVal.getBitWidth() == 1);

            JumpArg *jumpArg;
            if (constantIntVal.getValue() == 0)
              jumpArg = elseArg;
            else
              jumpArg = ifArg;

            auto jump = builder.create<inst::Jump>(oldExit.getRange(),
                                                   jumpArg->getAsState());
            return jump;
          } else if (auto sw = oldExit.dyn_cast<inst::Switch>()) {
            auto defaultArg = sw.getDefaultCase();
            llvm::SmallVector<JumpArg *> cases = llvm::map_to_vector(
                std::views::iota(0u, (unsigned)sw.getCaseSize()),
                [&](auto idx) { return sw.getCaseJumpArg(idx); });
            if (llvm::all_of(cases, [defaultArg](JumpArg *arg) {
                  return *arg == *defaultArg;
                })) {
              auto jump = builder.create<inst::Jump>(oldExit.getRange(),
                                                     defaultArg->getAsState());
              return jump;
            }

            auto value = sw.getValue();
            auto constantInst =
                value.getInstruction()->getDefiningInst<inst::Constant>();
            if (!constantInst)
              return nullptr;

            auto constantVal = constantInst.getValue();
            for (std::size_t i = 0; i < sw.getCaseSize(); ++i) {
              auto caseValue = sw.getCaseValue(i);
              auto caseConstant = caseValue.getInstruction()
                                      ->getDefiningInst<inst::Constant>()
                                      .getValue();

              if (constantVal == caseConstant) {
                auto jump = builder.create<inst::Jump>(
                    oldExit.getRange(), sw.getCaseJumpArg(i)->getAsState());
                return jump;
              }
            }
            auto jump = builder.create<inst::Jump>(oldExit.getRange(),
                                                   defaultArg->getAsState());
            return jump;
          }

          return nullptr;
        });
  }
  return changed ? PassResult::success() : PassResult::skip();
}

bool CFGEmpty::candidateCondition(Module *module, Block *block) const {
  if (block->getParentFunction()->getEntryBlock() == block)
    // Entry block can't be a candidate
    return false;

  Instruction firstInst = (*block->begin());
  if (auto exit = firstInst.dyn_cast<BlockExit>())
    return true;

  return false;
}

void CFGEmpty::connectReturn(Module *module, Block *pred, Block *succ) const {
  BlockExit oldExit = pred->getExit();

  module->replaceExit(
      oldExit,
      [retExit = succ->getExit().cast<inst::Return>()](
          IRBuilder &builder, BlockExit oldExit) -> BlockExit {
        return builder.create<inst::Return>(
            oldExit.getRange(),
            llvm::map_to_vector(retExit.getValues(),
                                [](Value op) -> Value { return op; }));
      });
}

void CFGEmpty::connectJump(Module *module, Block *pred, Block *succ) const {
  BlockExit oldExit = pred->getExit();

  module->replaceExit(oldExit,
                      [succ, jumpExit = succ->getExit().cast<inst::Jump>()](
                          IRBuilder &builder, BlockExit oldExit) -> BlockExit {
                        llvm::SmallVector<Attribute> attrs;
                        attrs.reserve(oldExit.getStorage()->getAttributeSize());

                        for (auto [idx, jumpArg] : llvm::enumerate(
                                 oldExit.getStorage()->getJumpArgs())) {
                          if (jumpArg->getBlock() == succ) {
                            oldExit.getStorage()->setJumpArg(
                                idx, jumpExit.getJumpArg()->getAsState());
                          }
                        }

                        return oldExit;
                      });
}

PassResult CFGEmpty::run(Module *module, Function *fun) {
  if (!fun->hasDefinition())
    return PassResult::skip();

  llvm::DenseSet<Block *> candidate;
  for (Block *block : *fun) {
    if (candidateCondition(module, block))
      candidate.insert(block);
  }

  bool changed = false;
  while (!candidate.empty()) {
    auto cand = *candidate.begin();
    candidate.erase(cand);

    auto exit = cand->getExit();
    auto preds = module->getPredecessors(cand);
    if (auto ret = exit.dyn_cast<inst::Return>()) {
      llvm::SmallVector<Block *> mergablePred;
      for (Block *pred : preds) {
        if (module->getSuccessors(pred).size() == 1)
          mergablePred.emplace_back(pred);
      }

      for (Block *pred : mergablePred) {
        changed = true;
        connectReturn(module, pred, cand);
        if (candidateCondition(module, pred))
          candidate.insert(pred);
      }
    } else if (auto jump = exit.dyn_cast<inst::Jump>()) {
      for (Block *pred : preds) {
        changed = true;
        connectJump(module, pred, cand);
        if (candidateCondition(module, pred))
          candidate.insert(pred);
      }
    }
  }

  return changed ? PassResult::success() : PassResult::skip();
}

Block *CFGMerge::find(Block *block) {
  return parent[block] == block ? block : (parent[block] = find(parent[block]));
}

void CFGMerge::setPar(Block *par, Block *child) { parent[child] = par; }

void CFGMerge::merge(Module *module, Block *child, Block *parent) const {
  auto jump = parent->getExit().cast<inst::Jump>();
  auto jumpArg = jump.getJumpArg();
  auto phiArg = jumpArg->getArgs();

  llvm::DenseMap<Value, Value> valueReplacement;

  unsigned idx = 0u;
  for (auto I = child->phiBegin(), E = child->phiEnd(); I != E; ++I, ++idx) {
    auto phi = (*I)->getDefiningInst<Phi>();
    Value arg = phiArg[idx];
    valueReplacement.try_emplace(phi, arg);
  }

  IRBuilder builder(module->getContext());
  builder.setInsertionPoint(parent->getLastTempInsertionPoint());
  for (auto I = child->tempBegin(), E = child->tempEnd(); I != E; ++I) {
    auto cloned = builder.clone(*I);
    for (auto [oldVal, newVal] :
         llvm::zip((*I)->getResults(), cloned->getResults())) {
      auto [_, inserted] = valueReplacement.try_emplace(oldVal, newVal);
      assert(inserted && "Value replacement should not have duplicates");
      (void)inserted;
    }
  }

  for (auto [oldV, newV] : valueReplacement)
    oldV.replaceWith(newV);

  module->replaceExit(parent->getExit(),
                      [&](IRBuilder &builder, BlockExit oldExit) -> BlockExit {
                        auto cloned =
                            builder.clone(child->getExit().getStorage());
                        return cloned->getDefiningInst<BlockExit>();
                      });
}

PassResult CFGMerge::run(Module *module, Function *fun) {
  if (!fun->hasDefinition())
    return PassResult::skip();

  for (Block *block : *fun) {
    parent[block] = block;
  }

  // pair means (child, parent)
  llvm::SmallVector<std::pair<Block *, Block *>> mergeCandPair;

  for (Block *block : *fun) {
    auto succs = module->getSuccessors(block);
    if (succs.size() != 1)
      continue;

    auto succ = *succs.begin();
    if (succ == block || succ == fun->getEntryBlock())
      continue; // skip self-loop and entry block

    auto exit = block->getExit();
    if (!exit.isa<inst::Jump>())
      continue;

    if (module->getPredecessors(succ).size() != 1)
      continue; // skip if the successor has multiple predecessors

    mergeCandPair.emplace_back(succ, block);
  }

  llvm::DenseSet<Block *> removedBlock;
  for (auto [child, parent] : mergeCandPair) {
    parent = find(parent);
    setPar(parent, child);
    merge(module, child, parent);
    removedBlock.insert(child);
  }

  for (Block *block : removedBlock) {
    block->dropReferences();
  }
  for (Block *block : removedBlock) {
    module->removeBlock(block);
  }

  return !mergeCandPair.empty() ? PassResult::success() : PassResult::skip();
}

void CFGReach::dfs(Module *module, Block *block) {
  if (visited.contains(block))
    return;

  visited.insert(block);
  for (Block *succ : module->getSuccessors(block)) {
    dfs(module, succ);
  }
}

PassResult CFGReach::run(Module *module, Function *fun) {
  if (!fun->hasDefinition())
    return PassResult::skip();

  dfs(module, fun->getEntryBlock());

  llvm::DenseSet<Block *> unreachableBlock;
  for (Block *block : *fun) {
    if (!visited.contains(block)) {
      unreachableBlock.insert(block);
    }
  }

  for (Block *block : unreachableBlock) {
    block->dropReferences();
  }
  for (Block *block : unreachableBlock) {
    module->removeBlock(block);
  }

  return !unreachableBlock.empty() ? PassResult::success() : PassResult::skip();
}

void registerSimplifyCFGPass() {
  registerPass<SimplifyCFG>();
  registerPass<CFGConstantPropagation>();
  registerPass<CFGEmpty>();
  registerPass<CFGMerge>();
  registerPass<CFGReach>();
}

} // namespace kecc::ir
