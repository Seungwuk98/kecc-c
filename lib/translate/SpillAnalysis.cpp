#include "kecc/translate/SpillAnalysis.h"
#include "kecc/ir/IRAnalyses.h"
#include "kecc/ir/IRInstructions.h"
#include "kecc/ir/Instruction.h"
#include "kecc/ir/WalkSupport.h"
#include "kecc/translate/InterferenceGraph.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include <cmath>

namespace kecc {

long double SpillCost::getSpillCost(LiveRange liveRange) const {
  return spillCostMap.at(liveRange);
}

// spill cost = Sum( use_def_number * 10 ^ loop_depth_of_block ) /
//                degree_of_live_range
void SpillCost::estimateSpillCost() {
  LiveRangeAnalysis *liveRangeAnalysis =
      module->getAnalysis<LiveRangeAnalysis>();
  assert(liveRangeAnalysis && "LiveRangeAnalysis must not be null");
  LivenessAnalysis *livenessAnalysis = module->getAnalysis<LivenessAnalysis>();
  assert(livenessAnalysis && "LivenessAnalysis must not be null");
  ir::LoopAnalysis *loopAnalysis = module->getAnalysis<ir::LoopAnalysis>();
  if (!loopAnalysis) {
    module->insertAnalysis(ir::LoopAnalysis::create(module));
    loopAnalysis = module->getAnalysis<ir::LoopAnalysis>();
  }

  SpillAnalysis *spillAnalysis = module->getAnalysis<SpillAnalysis>();

  // handle local variables
  for (ir::InstructionStorage *inst : *function->getAllocationBlock()) {
    ir::inst::LocalVariable localVar =
        inst->getDefiningInst<ir::inst::LocalVariable>();
    assert(localVar && "LocalVariable expected");
    LiveRange liveRange = liveRangeAnalysis->getLiveRange(function, localVar);
    if (spillAnalysis &&
        spillAnalysis->getSpillInfo().spilled.contains(liveRange)) {
      spillCostMap[liveRange] = std::numeric_limits<long double>::max();
    } else {
      spillCostMap[liveRange] = 1;
    }
  }

  for (ir::Block *block : *function) {
    auto loopDepth = loopAnalysis->getLoopDepth(block);
    auto depthValue = std::pow(SPILL_COST_BASE, loopDepth);

    for (ir::InstructionStorage *inst : *block) {
      auto results = inst->getResults();

      // handle restore
      inst->walk([&](const ir::Operand &op) -> ir::WalkResult {
        if (op.isConstant())
          return ir::WalkResult::advance();

        LiveRange liveRange;
        if (spillAnalysis &&
            spillAnalysis->getSpillInfo().restore.contains(&op)) {
          liveRange = spillAnalysis->getSpillInfo().restore.at(&op);
          spillCostMap[liveRange] = std::numeric_limits<long double>::max();
        } else {
          liveRange = liveRangeAnalysis->getLiveRange(function, op);
          if (spillCostMap.find(liveRange) == spillCostMap.end()) {
            spillCostMap[liveRange] = 0;
          }
          spillCostMap[liveRange] += depthValue;
        }
        return ir::WalkResult::advance();
      });

      // handle copies

      if (inst->getDefiningInst<ir::BlockExit>()) {
        for (const auto &[to, from] : liveRangeAnalysis->getCopyMap(block)) {
          auto toSpilled = spillAnalysis &&
                           spillAnalysis->getSpillInfo().spilled.contains(to);
          auto fromRestored =
              spillAnalysis &&
              spillAnalysis->getSpillInfo().restoreMemory.contains(from);

          if (!toSpilled) {
            if (spillCostMap.find(to) == spillCostMap.end()) {
              spillCostMap[to] = 0;
            }
            spillCostMap[to] += depthValue;
          } else {
            spillCostMap[to] = std::numeric_limits<long double>::max();
          }

          if (fromRestored)
            spillCostMap[from] = std::numeric_limits<long double>::max();
          else
            spillCostMap[from] += depthValue;
        }
      }

      for (ir::Value result : results) {
        auto liveRange = liveRangeAnalysis->getLiveRange(function, result);
        if (spillCostMap.find(liveRange) == spillCostMap.end()) {
          spillCostMap[liveRange] = 0;
        }
        // handle spilled
        if (spillAnalysis &&
            spillAnalysis->getSpillInfo().spilled.contains(liveRange))
          spillCostMap[liveRange] = std::numeric_limits<long double>::max();
        else
          spillCostMap[liveRange] += depthValue;
      }
    }
  }

  for (auto &[liveRange, cost] : spillCostMap) {
    auto degree = interfGraph->getDegree(liveRange);
    cost /= degree;
  }
}

void SpillCost::dump(llvm::raw_ostream &os) const {
  LiveRangeAnalysis *liveRangeAnalysis =
      module->getAnalysis<LiveRangeAnalysis>();
  assert(liveRangeAnalysis && "LiveRangeAnalysis must not be null");

  auto currIdMap = liveRangeAnalysis->getCurrLRIdMap();

  auto comparator = [&](LiveRange a, LiveRange b) {
    return currIdMap.at(a) < currIdMap.at(b);
  };

  auto sortedLiveRanges = llvm::map_to_vector(
      spillCostMap, [](const auto &pair) { return pair.first; });
  llvm::stable_sort(sortedLiveRanges, comparator);

  for (LiveRange lr : sortedLiveRanges) {
    auto cost = spillCostMap.at(lr);
    os << "L" << currIdMap[lr] << ": " << std::to_string(cost) << '\n';
  }
}

bool SpillAnalysis::trySpill(size_t iter) {
  bool spilled = false;
  for (ir::Function *func : *getModule()->getIR()) {
    auto count = 0;
    while (true) {
      if (iter >= 0 && count++ >= iter)
        break;

      auto funcSpilled = trySpill(func);
      if (!funcSpilled)
        break;
      spilled = true;
    }
  }
  return spilled;
}

bool SpillAnalysis::trySpill(ir::Function *func) {
  auto intResult = trySpill(func, as::RegisterType::Integer);
  auto floatResult = trySpill(func, as::RegisterType::FloatingPoint);
  return intResult || floatResult;
}

bool SpillAnalysis::trySpill(ir::Function *function, as::RegisterType regType) {
  auto interfGraph = getInterferenceGraph(function, regType);
  // find maximum clique using Maximum Cardinality Search
  MaximumCardinalitySearch *mcs = interfGraph->getMCS();
  const llvm::DenseSet<LiveRange> &maxClique = mcs->getMaxClique();

  LiveRangeAnalysis *liveRangeAnalysis =
      getModule()->getAnalysis<LiveRangeAnalysis>();
  assert(liveRangeAnalysis && "LiveRangeAnalysis must not be null");
  ir::LoopAnalysis *loopAnalysis = getModule()->getAnalysis<ir::LoopAnalysis>();
  if (!loopAnalysis) {
    getModule()->insertAnalysis(ir::LoopAnalysis::create(getModule()));
    loopAnalysis = getModule()->getAnalysis<ir::LoopAnalysis>();
  }

  llvm::ArrayRef<as::Register> avaiableRegs =
      translateContext->getRegistersForAllocate(
          interfGraph->isForFloatType() ? as::RegisterType::FloatingPoint
                                        : as::RegisterType::Integer);

  // Check if the number of available registers is sufficient
  if (avaiableRegs.size() >= maxClique.size())
    return false;

  // Calculate the number of spills needed
  auto spillCount = maxClique.size() - avaiableRegs.size();

  // Calculate spill costs for each live range in the maximum clique
  SpillCost spillCost(getModule(), function, interfGraph);

  auto currLRIdMap = liveRangeAnalysis->getCurrLRIdMap(getSpillInfo());
  auto comparator = [&](LiveRange a, LiveRange b) {
    size_t aCost = spillCost.getSpillCost(a);
    size_t bCost = spillCost.getSpillCost(b);
    if (aCost != bCost)
      return aCost < bCost; // Higher cost means lower priority

    return currLRIdMap.at(a) < currLRIdMap.at(b); // Use ID as tiebreaker
  };

  llvm::SmallVector<LiveRange> spillCandidates(maxClique.begin(),
                                               maxClique.end());
  llvm::stable_sort(spillCandidates, comparator);
  spillCandidates.erase(spillCandidates.begin() + spillCount,
                        spillCandidates.end());

  llvm::DenseSet<LiveRange> spilledLiveRanges(spillCandidates.begin(),
                                              spillCandidates.end());

  // Replace every values' live ranges with new live ranges
  SpillInfo newSpillInfo =
      liveRangeAnalysis->spill(function, spilledLiveRanges);

  spillInfo.insert(newSpillInfo);
  // Update liveness analysis
  auto newLivenessAnalysis = LivenessAnalysis::create(getModule(), spillInfo);
  getModule()->insertAnalysis(std::move(newLivenessAnalysis));

  // create a new interference graph
  auto newInterfGraph = InterferenceGraph::create(
      getModule(), function, interfGraph->isForFloatType());

  regType == as::RegisterType::Integer
      ? interfGraphMap[function].first = std::move(newInterfGraph)
      : interfGraphMap[function].second = std::move(newInterfGraph);

  return true;
}

void SpillAnalysis::spillFull() {
  for (ir::Function *function : *getModule()->getIR()) {
    if (!function->hasDefinition())
      continue;

    while (trySpill(function, as::RegisterType::Integer))
      ;
    while (trySpill(function, as::RegisterType::FloatingPoint))
      ;
  }
}

void SpillAnalysis::dumpInterferenceGraph(llvm::raw_ostream &os) const {
  auto liveRangeAnalysis = getModule()->getAnalysis<LiveRangeAnalysis>();
  assert(liveRangeAnalysis && "LiveRangeAnalysis must not be null");
  liveRangeAnalysis->dump(os, spillInfo);

  auto livenessAnalysis = getModule()->getAnalysis<LivenessAnalysis>();
  livenessAnalysis->dump(os);

  for (ir::Function *func : *getModule()->getIR()) {
    if (!func->hasDefinition())
      continue;

    os << "Interference graph for function: @" << func->getName() << "\n";
    os << "For integer live ranges:\n";
    auto intGraph = getInterferenceGraph(func, as::RegisterType::Integer);
    assert(intGraph && "Interference graph must not be null");
    intGraph->dump(os);

    os << "For floating-point live ranges:\n";
    auto floatGraph =
        getInterferenceGraph(func, as::RegisterType::FloatingPoint);
    assert(floatGraph && "Interference graph must not be null");
    floatGraph->dump(os);

    os << '\n';
  }
}

std::unique_ptr<SpillAnalysis>
SpillAnalysis::create(ir::Module *module, TranslateContext *translateContext) {
  llvm::DenseMap<ir::Function *, std::pair<std::unique_ptr<InterferenceGraph>,
                                           std::unique_ptr<InterferenceGraph>>>
      interfGraphMap;

  for (ir::Function *function : *module->getIR()) {
    if (!function->hasDefinition())
      continue;
    auto intGraph = InterferenceGraph::create(module, function, false);
    auto floatGraph = InterferenceGraph::create(module, function, true);
    interfGraphMap[function] = {std::move(intGraph), std::move(floatGraph)};
  }

  return std::unique_ptr<SpillAnalysis>(
      new SpillAnalysis(module, translateContext, std::move(interfGraphMap)));
}

} // namespace kecc
