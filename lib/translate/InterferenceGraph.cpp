#include "kecc/translate/InterferenceGraph.h"
#include "kecc/ir/IRAnalyses.h"
#include "kecc/ir/IRTypes.h"

namespace kecc {

struct InterferenceGraphBuilder {
  InterferenceGraphBuilder(ir::Module *module, ir::Function *func,
                           ir::LiveRangeAnalysis *liveRangeAnalysis,
                           ir::LivenessAnalysis *livenessAnalysis, bool isFloat)
      : module(module), func(func), liveRangeAnalysis(liveRangeAnalysis),
        livenessAnalysis(livenessAnalysis), isFloat(isFloat) {}

  void build();

  void insert(size_t lr1, size_t lr2);

  ir::Module *module;
  ir::Function *func;
  ir::LiveRangeAnalysis *liveRangeAnalysis;
  ir::LivenessAnalysis *livenessAnalysis;
  bool isFloat;
  std::map<size_t, std::set<size_t>> graph;
};

std::unique_ptr<InterferenceGraph> InterferenceGraph::create(ir::Module *module,
                                                             ir::Function *func,
                                                             bool isFloat) {
  auto *liveRangeAnalysis = module->getAnalysis<ir::LiveRangeAnalysis>();
  if (!liveRangeAnalysis) {
    auto analysis = ir::LiveRangeAnalysis::create(module);
    module->insertAnalysis(std::move(analysis));
    liveRangeAnalysis = module->getAnalysis<ir::LiveRangeAnalysis>();
  }

  auto *livenessAnalysis = module->getAnalysis<ir::LivenessAnalysis>();
  if (!livenessAnalysis) {
    auto analysis = ir::LivenessAnalysis::create(module);
    module->insertAnalysis(std::move(analysis));
    livenessAnalysis = module->getAnalysis<ir::LivenessAnalysis>();
  }

  InterferenceGraphBuilder builder(module, func, liveRangeAnalysis,
                                   livenessAnalysis, isFloat);
  builder.build();

  return std::unique_ptr<InterferenceGraph>(
      new InterferenceGraph(std::move(builder.graph)));
}

void InterferenceGraphBuilder::build() {
  for (ir::Block *block : *func) {
    std::set<size_t> liveNow;
    for (auto I = block->rbegin(), E = block->rend(); I != E; ++I) {
      ir::InstructionStorage *inst = *I;
      auto results = inst->getResults();
      for (ir::Value result : results) {
        size_t liveRange = liveRangeAnalysis->getLiveRange(func, result);
        liveNow.erase(liveRange);

        for (size_t otherLR : liveNow)
          insert(liveRange, otherLR);
      }

      for (ir::Value operand : inst->getOperands()) {
        if (operand.isConstant())
          continue;

        size_t liveRange = liveRangeAnalysis->getLiveRange(func, operand);
        liveNow.insert(liveRange);
      }
    }
  }
}

void InterferenceGraphBuilder::insert(size_t lr1, size_t lr2) {
  auto lr1T = liveRangeAnalysis->getLiveRangeType(func, lr1);
  auto lr2T = liveRangeAnalysis->getLiveRangeType(func, lr2);
  auto lr1IsFloat = lr1T.isa<ir::FloatT>();
  auto lr2IsFloat = lr2T.isa<ir::FloatT>();

  if (isFloat && lr1IsFloat && lr2IsFloat) {
    graph[lr1].insert(lr2);
    graph[lr2].insert(lr1);
  } else if (!isFloat && !lr1IsFloat && !lr2IsFloat) {
    graph[lr1].insert(lr2);
    graph[lr2].insert(lr1);
  }
}

} // namespace kecc
