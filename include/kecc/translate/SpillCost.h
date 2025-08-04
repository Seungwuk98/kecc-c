#ifndef KECC_TRANSLATE_SPILL_COST_H
#define KECC_TRANSLATE_SPILL_COST_H

#include "kecc/ir/IR.h"
#include "kecc/ir/IRAnalyses.h"
#include "kecc/translate/InterferenceGraph.h"

namespace kecc {

static constexpr long double SPILL_COST_BASE = 10;

class SpillCost {
public:
  SpillCost(ir::Module *module, ir::Function *function,
            ir::LiveRangeAnalysis *liveRangeAnalysis,
            ir::LoopAnalysis *loopAnalysis, InterferenceGraph *interfGraph)
      : module(module), function(function),
        liveRangeAnalysis(liveRangeAnalysis), loopAnalysis(loopAnalysis),
        interfGraph(interfGraph) {
    assert(module && "Module must not be null");
    assert(function && "Function must not be null");
    assert(liveRangeAnalysis && "LiveRangeAnalysis must not be null");
    assert(loopAnalysis && "LoopAnalysis must not be null");
    estimateSpillCost();
  }

  long double getSpillCost(size_t liveRange) const;

private:
  void estimateSpillCost();

  ir::Module *module;
  ir::Function *function;
  ir::LiveRangeAnalysis *liveRangeAnalysis;
  ir::LoopAnalysis *loopAnalysis;
  InterferenceGraph *interfGraph;
  llvm::DenseMap<size_t, long double> spillCostMap;
};

} // namespace kecc

#endif // KECC_TRANSLATE_SPILL_COST_H
