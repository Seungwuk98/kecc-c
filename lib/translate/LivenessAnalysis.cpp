#include "kecc/ir/IRAnalyses.h"
#include "kecc/ir/IRInstructions.h"
#include "kecc/ir/Instruction.h"
#include "kecc/ir/Value.h"
#include "kecc/translate/LiveRangeAnalyses.h"

DEFINE_KECC_TYPE_ID(kecc::LivenessAnalysis)

namespace kecc {

struct LivenessAnalysisBuilder {
  LivenessAnalysisBuilder(LiveRangeAnalysis *analysis,
                          ir::VisitOrderAnalysis *visitOrderAnalysis,
                          const SpillInfo &spill)
      : liveRangeAnalysis(analysis), visitOrderAnalysis(visitOrderAnalysis),
        spill(spill) {}

  void build(ir::Module *module);

  LiveRangeAnalysis *liveRangeAnalysis;
  ir::VisitOrderAnalysis *visitOrderAnalysis;
  llvm::DenseMap<ir::Block *, llvm::DenseSet<LiveRange>> liveOut;
  llvm::DenseMap<ir::Block *, llvm::DenseSet<LiveRange>> uevar;
  llvm::DenseMap<ir::Block *, llvm::DenseSet<LiveRange>> varKill;
  const SpillInfo &spill;
};

namespace {

static llvm::DenseSet<LiveRange> sub(const llvm::DenseSet<LiveRange> &a,
                                     const llvm::DenseSet<LiveRange> &b) {
  llvm::DenseSet<LiveRange> result;
  for (LiveRange elem : a)
    if (!b.contains(elem))
      result.insert(elem);
  return result;
}

static llvm::DenseSet<LiveRange> unionSet(const llvm::DenseSet<LiveRange> &a,
                                          const llvm::DenseSet<LiveRange> &b) {
  llvm::DenseSet<LiveRange> result = a;
  result.insert(b.begin(), b.end());
  return result;
}

} // namespace

void LivenessAnalysisBuilder::build(ir::Module *module) {
  for (ir::Function *func : *module->getIR()) {
    auto order = visitOrderAnalysis->getReversePreOrder(func);
    for (ir::Block *block : order) {
      if (block == func->getEntryBlock()) {
        // entry phi instruction is a definition
        for (auto I = block->phiBegin(), E = block->phiEnd(); I != E; ++I) {
          ir::Phi phi = (*I)->getDefiningInst<ir::Phi>();
          assert(phi && "Phi instruction expected");
          LiveRange lr = liveRangeAnalysis->getLiveRange(func, phi);
          varKill[block].insert(lr);
        }
      }
      for (auto I = block->tempBegin(), E = block->end(); I != E; ++I) {
        ir::InstructionStorage *inst = *I;

        if (inst->getDefiningInst<ir::BlockExit>()) {
          for (const auto &[to, from] : liveRangeAnalysis->getCopyMap(block)) {
            if (spill.restoreMemory.contains(from))
              varKill[block].insert(from);
            else if (!varKill[block].contains(from))
              uevar[block].insert(from);

            // if `to` is a spilled value, this copy does not affect the
            // register
            if (!spill.spilled.contains(to))
              varKill[block].insert(to);
          }
        }

        auto results = inst->getResults();

        for (const ir::Operand &operand : inst->getOperands()) {
          if (operand.isConstant())
            continue;

          // If the operand is a restored value, we need to insert it into
          // varKill
          LiveRange liveRange;
          if (auto it = spill.restore.find(&operand);
              it != spill.restore.end()) {
            liveRange = it->getSecond();
            varKill[block].insert(liveRange);
          } else {
            liveRange = liveRangeAnalysis->getLiveRange(func, operand);
          }

          if (!varKill[block].contains(liveRange))
            uevar[block].insert(liveRange);
        }

        for (ir::Value result : results) {
          LiveRange liveRange = liveRangeAnalysis->getLiveRange(func, result);
          varKill[block].insert(liveRange);
        }
        // We need to handle spill values but the values are already inserted
        // in varKill, so we can skip them here.
      }
    }
  }

  LiveRangeAnalysis *liveRangeAnalysis =
      module->getAnalysis<LiveRangeAnalysis>();
  auto currLRIdMap = liveRangeAnalysis->getCurrLRIdMap(spill);

  for (ir::Function *func : *module->getIR()) {
    for (ir::Block *block : *func) {
      llvm::errs() << "block b" << block->getId() << ":\n";
      const auto &uevarSet = uevar[block];
      const auto &varKillSet = varKill[block];
      for (LiveRange lr : uevarSet) {
        llvm::errs() << "uevar: " << currLRIdMap.at(lr) << '\n';
      }
      for (LiveRange lr : varKillSet) {
        llvm::errs() << "varKill: " << currLRIdMap.at(lr) << '\n';
      }
    }
  }

  for (ir::Function *func : *module->getIR()) {
    auto order = visitOrderAnalysis->getReversePreOrder(func);
    bool changed;
    do {
      changed = false;
      for (ir::Block *block : order) {
        llvm::DenseSet<LiveRange> currLiveOut;
        auto succs = module->getSuccessors(block);
        for (ir::Block *succ : succs) {
          const auto &succLiveOut = liveOut[succ];
          const auto &succUevar = uevar[succ];
          const auto &succVarKill = varKill[succ];
          currLiveOut = unionSet(
              currLiveOut, unionSet(succUevar, sub(succLiveOut, succVarKill)));
        }

        if (liveOut[block] != currLiveOut) {
          changed = true;
          liveOut[block] = currLiveOut;
        }
      }
    } while (changed);

    // we need to calculate liveOut for allocation block
    ir::Block *allocBlock = func->getAllocationBlock();
    ir::Block *entryBlock = func->getEntryBlock();
    // only a successor of allocation block is the entry block
    auto allocLiveOut = unionSet(uevar[entryBlock],
                                 sub(liveOut[entryBlock], varKill[entryBlock]));
    if (liveOut[allocBlock] != allocLiveOut) {
      changed = true;
      liveOut[allocBlock] = allocLiveOut;
    }
  }
}

const llvm::DenseSet<LiveRange> &
LivenessAnalysis::getLiveVars(ir::Block *block) const {
  return liveOut.at(block);
}

std::unique_ptr<LivenessAnalysis>
LivenessAnalysis::create(ir::Module *module, const SpillInfo &spill) {
  auto *liveRangeAnalysis = module->getAnalysis<LiveRangeAnalysis>();
  if (!liveRangeAnalysis) {
    auto analysis = LiveRangeAnalysis::create(module);
    module->insertAnalysis(std::move(analysis));
    liveRangeAnalysis = module->getAnalysis<LiveRangeAnalysis>();
  }

  auto *visitOrderAnalysis = module->getAnalysis<ir::VisitOrderAnalysis>();
  if (!visitOrderAnalysis) {
    auto analysis = ir::VisitOrderAnalysis::create(module);
    module->insertAnalysis(std::move(analysis));
    visitOrderAnalysis = module->getAnalysis<ir::VisitOrderAnalysis>();
  }

  LivenessAnalysisBuilder builder(liveRangeAnalysis, visitOrderAnalysis, spill);
  builder.build(module);
  return std::unique_ptr<LivenessAnalysis>(
      new LivenessAnalysis(module, std::move(builder.liveOut)));
}

void LivenessAnalysis::dump(llvm::raw_ostream &os) const {
  os << "Liveness Analysis dump:";

  auto *liveRangeAnalysis = getModule()->getAnalysis<LiveRangeAnalysis>();
  assert(liveRangeAnalysis && "Liveness analysis requires live range analysis");
  auto currLRIdMap = liveRangeAnalysis->getCurrLRIdMap();

  for (ir::Function *func : *getModule()->getIR()) {
    if (!func->hasDefinition())
      continue;
    os << "\nFunction: @" << func->getName() << "\n";
    for (ir::Block *block : *func) {
      os << "  block b" << block->getId() << ":";
      const auto &liveVars = getLiveVars(block);
      llvm::SmallVector<LiveRange> sortedLiveVars(liveVars.begin(),
                                                  liveVars.end());
      llvm::stable_sort(sortedLiveVars, [&](LiveRange a, LiveRange b) {
        return currLRIdMap.at(a) < currLRIdMap.at(b);
      });

      if (liveVars.empty())
        os << " <empty>\n";
      else {
        os << ' ';
        for (auto I = sortedLiveVars.begin(), E = sortedLiveVars.end(); I != E;
             ++I) {
          if (I != sortedLiveVars.begin())
            os << ", ";
          os << 'L' << currLRIdMap[*I];
        }
        os << '\n';
      }
    }
  }
}

} // namespace kecc
