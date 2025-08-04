#include "kecc/translate/SpillCost.h"
#include "kecc/ir/Instruction.h"
#include "kecc/ir/WalkSupport.h"
#include <cmath>

namespace kecc {

long double SpillCost::getSpillCost(size_t liveRange) const {
  return spillCostMap.at(liveRange);
}

// spill cost = Sum( use_def_number * 10 ^ loop_depth_of_block ) /
//                degree_of_live_range
void SpillCost::estimateSpillCost() {
  for (ir::Block *block : *function) {
    auto loopDepth = loopAnalysis->getLoopDepth(block);
    auto depthValue = std::pow(SPILL_COST_BASE, loopDepth);

    for (ir::InstructionStorage *inst : *block) {
      auto results = inst->getResults();
      for (ir::Value result : results) {
        auto liveRange = liveRangeAnalysis->getLiveRange(function, result);
        if (spillCostMap.find(liveRange) == spillCostMap.end()) {
          spillCostMap[liveRange] = 0;
        }
        spillCostMap[liveRange] += depthValue;
      }

      inst->walk([&](const ir::Operand &op) -> ir::WalkResult {
        if (op.isConstant())
          return ir::WalkResult::advance();

        auto liveRange = liveRangeAnalysis->getLiveRange(function, op);
        if (spillCostMap.find(liveRange) == spillCostMap.end()) {
          spillCostMap[liveRange] = 0;
        }
        spillCostMap[liveRange] += depthValue;
        return ir::WalkResult::advance();
      });
    }
  }

  for (auto &[liveRange, cost] : spillCostMap) {
    auto degree = interfGraph->getDegree(liveRange);
    cost /= degree;
  }
}

} // namespace kecc
