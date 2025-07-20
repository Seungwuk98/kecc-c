#include "kecc/ir/IRAttributes.h"
#include "kecc/ir/IRInstructions.h"
#include "kecc/ir/IRTransforms.h"
#include "kecc/ir/Instruction.h"
#include "kecc/ir/Pass.h"

namespace kecc::ir {

class FunctionDeadCode {
public:
  FunctionDeadCode(Module *module, Function *function, inst::Constant *unitV)
      : module(module), function(function), unitV(unitV) {}

  PassResult run();

  void removePhi(Block *block, llvm::ArrayRef<Phi> phis);

private:
  Module *module;
  Function *function;
  inst::Constant *unitV;
};

PassResult DeadCode::run(Module *module) {
  inst::Constant unitV;
  for (Instruction inst : *module->getIR()->getConstantBlock()) {
    auto constInst = inst.cast<inst::Constant>();
    auto constValue = constInst.getValue();

    if (constValue == ConstantUnitAttr::get(module->getContext())) {
      unitV = constInst;
      break;
    }
  }

  if (!unitV) {
    IRBuilder builder(module->getContext());
    builder.setInsertionPoint(module->getIR()->getConstantBlock());
    unitV = builder.create<inst::Constant>(
        {}, ConstantUnitAttr::get(module->getContext()));
  }

  PassResult result;
  do {
    result = PassResult::skip();
    for (Function *function : *module->getIR()) {
      FunctionDeadCode functionDeadCode(module, function, &unitV);
      PassResult funcResult = functionDeadCode.run();
      if (funcResult.isFailure())
        return result;
      if (funcResult.isSuccess())
        result = funcResult;
    }
  } while (result.isSuccess());

  // Constant elimination
  llvm::SmallVector<InstructionStorage *, 8> toDelete;
  for (InstructionStorage *inst : *module->getIR()->getConstantBlock()) {
    auto constInst = inst->getDefiningInst<inst::Constant>();
    if (!constInst.getResult().hasUses())
      toDelete.emplace_back(inst);
  }

  for (InstructionStorage *inst : toDelete)
    module->getIR()->getConstantBlock()->remove(inst);

  return toDelete.empty() ? PassResult::skip() : result;
}

void FunctionDeadCode::removePhi(Block *block, llvm::ArrayRef<Phi> phis) {
  llvm::DenseSet<Phi> phiSet(phis.begin(), phis.end());

  llvm::DenseSet<size_t> phiIndices;
  phiIndices.reserve(phis.size());

  size_t phiIdx = 0;
  for (auto I = block->phiBegin(), E = block->phiEnd(); I != E; ++I, ++phiIdx) {
    if (phiSet.contains(*I))
      phiIndices.insert(phiIdx);
  }

  auto preds = module->getPredecessors(block);

  for (Block *pred : preds) {
    module->replaceExit(
        pred->getExit(),
        [&](IRBuilder &builder, BlockExit oldExit) -> BlockExit {
          for (auto [idx, jumpArg] :
               llvm::enumerate(oldExit.getStorage()->getJumpArgs())) {
            if (jumpArg->getBlock() == block) {
              auto argState = jumpArg->getAsState();
              llvm::SmallVector<Value> newArgs;
              newArgs.reserve(argState.getArgs().size() - phiIndices.size());
              for (size_t i = 0; i < argState.getArgs().size(); ++i) {
                if (phiIndices.contains(i))
                  continue;
                newArgs.emplace_back(argState.getArgs()[i]);
              }
              argState.setArgs(newArgs);
              oldExit.getStorage()->setJumpArg(idx, argState);
            }
          }
          return oldExit;
        });
  }

  for (Phi phi : phis) {
    module->removeInst(phi.getStorage());
  }
}

PassResult FunctionDeadCode::run() {
  PassResult result = PassResult::skip();
  llvm::SmallVector<InstructionStorage *, 8> toDeleteInst;
  llvm::DenseMap<Block *, llvm::SmallVector<Phi, 4>> toDeletePhis;

  for (InstructionStorage *inst : *function->getAllocationBlock()) {
    auto allocInst = inst->getDefiningInst<inst::LocalVariable>();
    if (!allocInst.getResult().hasUses())
      toDeleteInst.emplace_back(inst);
  }

  assert(
      function->getUnresolvedBlock()->empty() &&
      "Unresolved block should be empty before running dead code elimination");

  for (Block *block : *function) {
    if (block != function->getEntryBlock()) {
      for (auto I = block->phiBegin(); I != block->phiEnd(); ++I) {
        Phi phi = (*I)->getDefiningInst<Phi>();
        if (!phi.getResult().hasUses()) {
          toDeletePhis[block].emplace_back(phi);
        }
      }
    }

    for (auto I = block->tempBegin(), E = block->tempEnd(); I != E; ++I) {
      Instruction inst = *I;
      if (inst.hasTrait<SideEffect>())
        continue;

      auto results = inst.getStorage()->getResults();
      bool hasUses = llvm::all_of(results, [](Value result) {
        return result.hasUses() && !result.getType().isa<UnitT>();
      });

      if (!hasUses)
        toDeleteInst.emplace_back(inst.getStorage());
    }
  }

  for (InstructionStorage *inst : toDeleteInst) {
    for (Value result : inst->getResults()) {
      if (result.hasUses()) {
        assert(result.getType().isa<UnitT>() &&
               "Result should be a unit type if it has uses");

        if (!unitV) {
          IRBuilder builder(module->getContext());
          builder.setInsertionPoint(module->getIR()->getConstantBlock());
          *unitV = builder.create<inst::Constant>(
              {}, ConstantUnitAttr::get(module->getContext()));
        }

        result.replaceWith(*unitV);
      }
    }

    module->removeInst(inst);
  }

  for (auto &[block, phis] : toDeletePhis) {
    assert(!phis.empty() && "There should be phis to delete in the block");
    removePhi(block, phis);
  }

  return (toDeleteInst.empty() && toDeletePhis.empty()) ? PassResult::skip()
                                                        : PassResult::success();
}

} // namespace kecc::ir
