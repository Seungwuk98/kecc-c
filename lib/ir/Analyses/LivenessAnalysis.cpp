#include "kecc/ir/IRAnalyses.h"
#include "kecc/ir/Instruction.h"

DEFINE_KECC_TYPE_ID(kecc::ir::LivenessAnalysis)

namespace kecc::ir {

struct LivenessAnalysisBuilder {
  LivenessAnalysisBuilder(LiveRangeAnalysis *analysis,
                          VisitOrderAnalysis *visitOrderAnalysis)
      : liveRangeAnalysis(analysis), visitOrderAnalysis(visitOrderAnalysis) {}

  void build(Module *module);

  LiveRangeAnalysis *liveRangeAnalysis;
  VisitOrderAnalysis *visitOrderAnalysis;
  llvm::DenseMap<Block *, std::set<size_t>> liveOut;
  llvm::DenseMap<Block *, std::set<size_t>> uevar;
  llvm::DenseMap<Block *, std::set<size_t>> varKill;
};

namespace {

static std::set<size_t> sub(const std::set<size_t> &a,
                            const std::set<size_t> &b) {
  std::set<size_t> result;
  for (size_t elem : a)
    if (!b.contains(elem))
      result.insert(elem);
  return result;
}

static std::set<size_t> unionSet(const std::set<size_t> &a,
                                 const std::set<size_t> &b) {
  std::set<size_t> result = a;
  result.insert(b.begin(), b.end());
  return result;
}

} // namespace

void LivenessAnalysisBuilder::build(Module *module) {
  for (Function *func : *module->getIR()) {
    auto order = visitOrderAnalysis->getReversePreOrder(func);
    for (Block *block : order) {
      for (auto I = block->rbegin(), E = block->rend(); I != E; ++I) {
        InstructionStorage *inst = *I;
        auto results = inst->getResults();
        for (Value result : results) {
          size_t liveRange = liveRangeAnalysis->getLiveRange(func, result);
          varKill[block].insert(liveRange);
        }

        for (Value operand : inst->getOperands()) {
          if (operand.isConstant())
            continue;

          size_t liveRange = liveRangeAnalysis->getLiveRange(func, operand);
          uevar[block].insert(liveRange);
        }
      }
    }
  }

  for (Function *func : *module->getIR()) {
    auto order = visitOrderAnalysis->getReversePreOrder(func);
    bool changed;
    do {
      changed = false;
      for (Block *block : order) {
        std::set<size_t> currLiveOut;
        auto succs = module->getSuccessors(block);
        for (Block *succ : succs) {
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
  }
}

const std::set<size_t> &LivenessAnalysis::getLiveVars(Block *block) const {
  return liveOut.at(block);
}

std::unique_ptr<LivenessAnalysis> LivenessAnalysis::create(Module *module) {
  auto *liveRangeAnalysis = module->getAnalysis<LiveRangeAnalysis>();
  if (!liveRangeAnalysis) {
    auto analysis = LiveRangeAnalysis::create(module);
    module->insertAnalysis(std::move(analysis));
    liveRangeAnalysis = module->getAnalysis<LiveRangeAnalysis>();
  }

  auto *visitOrderAnalysis = module->getAnalysis<VisitOrderAnalysis>();
  if (!visitOrderAnalysis) {
    auto analysis = VisitOrderAnalysis::create(module);
    module->insertAnalysis(std::move(analysis));
    visitOrderAnalysis = module->getAnalysis<VisitOrderAnalysis>();
  }

  LivenessAnalysisBuilder builder(liveRangeAnalysis, visitOrderAnalysis);
  builder.build(module);
  return std::unique_ptr<LivenessAnalysis>(
      new LivenessAnalysis(module, std::move(builder.liveOut)));
}

void LivenessAnalysis::dump(llvm::raw_ostream &os) const {
  os << "Liveness Analysis dump:";

  for (Function *func : *getModule()->getIR()) {
    if (!func->hasDefinition())
      continue;
    os << "\nFunction: @" << func->getName() << "\n";
    for (Block *block : *func) {
      os << "  block b" << block->getId() << ":";
      const auto &liveVars = getLiveVars(block);
      if (liveVars.empty())
        os << " <empty>\n";
      else {
        os << ' ';
        for (auto I = liveVars.begin(), E = liveVars.end(); I != E; ++I) {
          if (I != liveVars.begin())
            os << ", ";
          os << 'L' << *I;
        }
        os << '\n';
      }
    }
  }
}

} // namespace kecc::ir
