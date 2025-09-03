#include "kecc/translate/InterferenceGraph.h"
#include "kecc/ir/IRInstructions.h"
#include "kecc/ir/IRTypes.h"
#include "kecc/ir/Instruction.h"
#include "kecc/translate/LiveRangeAnalyses.h"
#include "kecc/translate/SpillAnalysis.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include <queue>

namespace kecc {

size_t InterferenceGraph::getDegree(LiveRange liveRange) const {
  auto it = graph.find(liveRange);
  if (it != graph.end()) {
    return it->second.size();
  }
  return 0;
}

struct InterferenceGraphBuilder {
  InterferenceGraphBuilder(ir::Module *module, ir::Function *func,
                           LiveRangeAnalysis *liveRangeAnalysis,
                           LivenessAnalysis *livenessAnalysis, bool isFloat)
      : module(module), func(func), liveRangeAnalysis(liveRangeAnalysis),
        livenessAnalysis(livenessAnalysis), isFloat(isFloat) {}

  void build();

  void insert(LiveRange lr1, LiveRange lr2);

  ir::Module *module;
  ir::Function *func;
  LiveRangeAnalysis *liveRangeAnalysis;
  LivenessAnalysis *livenessAnalysis;
  bool isFloat;
  llvm::DenseMap<LiveRange, llvm::DenseSet<LiveRange>> graph;
};

void InterferenceGraphBuilder::build() {
  SpillAnalysis *spillAnalysis = module->getAnalysis<SpillAnalysis>();

  // special case for function argument
  // All function arguments are interfering with each other

  {
    llvm::SmallVector<LiveRange> funcArgs;
    for (ir::InstructionStorage *inst : *func->getEntryBlock()) {
      auto funcArg = inst->getDefiningInst<ir::inst::FunctionArgument>();
      if (!funcArg)
        break;
      auto liveRange = liveRangeAnalysis->getLiveRange(func, funcArg);
      for (LiveRange otherLR : funcArgs) {
        insert(liveRange, otherLR);
      }
      funcArgs.emplace_back(liveRange);
    }
  }

  for (ir::Block *block : *func) {
    llvm::DenseSet<LiveRange> liveNow = livenessAnalysis->getLiveVars(block);

    for (auto I = block->rbegin(), E = block->rend(); I != E; ++I) {
      ir::InstructionStorage *inst = *I;
      if (inst->getDefiningInst<ir::Phi>())
        break;

      auto results = inst->getResults();
      auto liveRanges = llvm::map_to_vector(results, [&](ir::Value result) {
        return liveRangeAnalysis->getLiveRange(func, result);
      });

      if (spillAnalysis) {
        // spill instructions
        for (LiveRange lr : liveRanges) {
          if (spillAnalysis->getSpillInfo().spilled.contains(lr))
            liveNow.insert(lr);
        }
      }

      for (LiveRange lr : liveRanges) {
        for (LiveRange otherLR : liveNow)
          if (otherLR != lr)
            insert(lr, otherLR);
        liveNow.erase(lr);
      }

      // interference with results
      for (auto i = 0u; i < liveRanges.size(); ++i) {
        for (auto j = i + 1; j < liveRanges.size(); ++j) {
          assert(liveRanges[i] != liveRanges[j] &&
                 "Live ranges should not be equal");
          insert(liveRanges[i], liveRanges[j]);
        }
      }

      llvm::SmallVector<LiveRange> restored;
      for (const ir::Operand &operand : inst->getOperands()) {
        if (operand.isConstant())
          continue;

        LiveRange liveRange;
        if (spillAnalysis &&
            spillAnalysis->getSpillInfo().restore.contains(&operand)) {
          liveRange = spillAnalysis->getSpillInfo().restore.at(&operand);
          restored.emplace_back(liveRange);
        } else {
          liveRange = liveRangeAnalysis->getLiveRange(func, operand);
          if (spillAnalysis &&
              spillAnalysis->getSpillInfo().spilled.contains(liveRange)) {
            assert(!inst->hasTrait<ir::CallLike>() &&
                   "Spilled live range should not be used in call-like "
                   "instructions");
            continue;
          }
        }
        liveNow.insert(liveRange);
      }

      if (auto exit = inst->getDefiningInst<ir::BlockExit>()) {
        // handle copies
        auto copyMap = liveRangeAnalysis->getCopyMap(block);
        for (const auto &[to, from] : llvm::reverse(copyMap)) {
          auto toSpilled = spillAnalysis &&
                           spillAnalysis->getSpillInfo().spilled.contains(to);
          auto fromRestored =
              spillAnalysis &&
              spillAnalysis->getSpillInfo().restoreMemory.contains(from);

          if (!toSpilled) {
            for (LiveRange otherLR : liveNow) {
              if (otherLR != to && otherLR != from)
                insert(to, otherLR);
            }
            liveNow.erase(to);
          }
          liveNow.insert(from);
          if (fromRestored) {
            for (LiveRange otherLR : liveNow) {
              if (otherLR != from)
                insert(from, otherLR);
            }
          }
        }
      }

      for (LiveRange restoredLR : restored) {
        for (LiveRange otherLR : liveNow) {
          if (otherLR != restoredLR)
            insert(restoredLR, otherLR);
        }
        liveNow.erase(restoredLR);
      }
    }
  }

  // handle allocation block
  auto *allocBlock = func->getAllocationBlock();
  auto liveNow = livenessAnalysis->getLiveVars(allocBlock);
  for (auto I = allocBlock->rbegin(), E = allocBlock->rend(); I != E; ++I) {
    ir::InstructionStorage *inst = *I;
    ir::inst::LocalVariable localVar =
        inst->getDefiningInst<ir::inst::LocalVariable>();
    assert(localVar && "Expected LocalVariable instruction");
    LiveRange liveRange = liveRangeAnalysis->getLiveRange(func, localVar);
    if (spillAnalysis &&
        spillAnalysis->getSpillInfo().spilled.contains(liveRange)) {
      liveNow.insert(liveRange);
    }

    for (LiveRange otherLR : liveNow)
      if (otherLR != liveRange) {
        insert(liveRange, otherLR);
      }

    liveNow.erase(liveRange);
  }
}

void InterferenceGraphBuilder::insert(LiveRange lr1, LiveRange lr2) {
  auto lr1T = lr1.getType();
  auto lr2T = lr2.getType();
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

std::unique_ptr<InterferenceGraph> InterferenceGraph::create(ir::Module *module,
                                                             ir::Function *func,
                                                             bool isFloat) {
  auto *liveRangeAnalysis =
      module->getOrCreateAnalysis<LiveRangeAnalysis>(module);
  auto *livenessAnalysis =
      module->getOrCreateAnalysis<LivenessAnalysis>(module);

  InterferenceGraphBuilder builder(module, func, liveRangeAnalysis,
                                   livenessAnalysis, isFloat);
  builder.build();

  return std::unique_ptr<InterferenceGraph>(
      new InterferenceGraph(module, func, isFloat, std::move(builder.graph)));
}

void InterferenceGraph::update() {
  auto *liveRangeAnalysis = module->getAnalysis<LiveRangeAnalysis>();
  if (!liveRangeAnalysis) {
    auto analysis = LiveRangeAnalysis::create(module);
    module->insertAnalysis(std::move(analysis));
    liveRangeAnalysis = module->getAnalysis<LiveRangeAnalysis>();
  }

  auto *livenessAnalysis = module->getAnalysis<LivenessAnalysis>();
  if (!livenessAnalysis) {
    auto analysis = LivenessAnalysis::create(module);
    module->insertAnalysis(std::move(analysis));
    livenessAnalysis = module->getAnalysis<LivenessAnalysis>();
  }

  InterferenceGraphBuilder builder(module, function, liveRangeAnalysis,
                                   livenessAnalysis, isFloat);
  builder.build();
  graph = std::move(builder.graph);
}

InterferenceGraph::InterferenceGraph(
    ir::Module *module, ir::Function *function, bool isFloat,
    llvm::DenseMap<LiveRange, llvm::DenseSet<LiveRange>> graph)
    : module(module), function(function), isFloat(isFloat),
      graph(std::move(graph)), graphColoring(nullptr) {
  mcs.reset(new MaximumCardinalitySearch(this));
}
InterferenceGraph::~InterferenceGraph() = default;

GraphColoring *InterferenceGraph::coloring() {
  graphColoring = std::make_unique<GraphColoring>(this);
  return getGraphColoring();
}

void InterferenceGraph::dump(llvm::raw_ostream &os) const {
  LiveRangeAnalysis *liveRangeAnalysis =
      module->getAnalysis<LiveRangeAnalysis>();
  assert(liveRangeAnalysis && "LiveRangeAnalysis must not be null");

  SpillAnalysis *spillAnalysis = module->getAnalysis<SpillAnalysis>();

  auto currLRIdMap = liveRangeAnalysis->getCurrLRIdMap(
      spillAnalysis ? spillAnalysis->getSpillInfo() : SpillInfo());

  auto keys =
      llvm::map_to_vector(graph, [](const auto &pair) { return pair.first; });

  auto comparator = [&](LiveRange a, LiveRange b) {
    auto aId = currLRIdMap.at(a);
    auto bId = currLRIdMap.at(b);
    return aId < bId; // sort by id
  };

  llvm::sort(keys, comparator);

  for (auto liveRange : keys) {
    os << "L" << currLRIdMap[liveRange] << ": ";
    const auto &neighbors = graph.at(liveRange);
    if (neighbors.empty())
      os << "<empty>";
    else {
      llvm::SmallVector<LiveRange> sortedNeighbors(neighbors.begin(),
                                                   neighbors.end());
      llvm::sort(sortedNeighbors, comparator);
      for (auto I = sortedNeighbors.begin(), E = sortedNeighbors.end(); I != E;
           ++I) {
        if (I != sortedNeighbors.begin())
          os << ", ";
        os << "L" << currLRIdMap[*I];
      }
    }
    os << "\n";
  }

  if (mcs) {
    mcs->dump(os);
  }
}

void MaximumCardinalitySearch::init() {
  auto *liveRangeAnalysis =
      interferenceGraph->module->getAnalysis<LiveRangeAnalysis>();
  assert(liveRangeAnalysis &&
         "LiveRangeAnalysis must be available before MaximumCardinalitySearch");

  auto *spillAnalysis = interferenceGraph->module->getAnalysis<SpillAnalysis>();

  liveRangeIdMap = liveRangeAnalysis->getCurrLRIdMap(
      spillAnalysis ? spillAnalysis->getSpillInfo() : SpillInfo());
}

void MaximumCardinalitySearch::search() {
  using PQKey = std::pair<size_t, LiveRange>;

  auto comparator = [&](const PQKey &a, const PQKey &b) {
    if (a.first != b.first)
      return a.first < b.first; // max heap
    return liveRangeIdMap.at(a.second) >
           liveRangeIdMap.at(b.second); // tie-break by id
  };

  std::priority_queue<PQKey, std::vector<PQKey>, decltype(comparator)> pq(
      comparator);

  for (const auto &[lr, _] : interferenceGraph->graph) {
    clique[lr];
    pq.push({0, lr});
  }

  while (!pq.empty()) {
    auto [cardinality, liveRange] = pq.top();
    pq.pop();
    if (cardinality != clique.at(liveRange).size() ||
        simplicialElimOrder.contains(liveRange))
      continue;

    simplicialElimOrder.insert(liveRange);

    for (const auto &neighbor : interferenceGraph->graph[liveRange]) {
      if (simplicialElimOrder.contains(neighbor))
        continue;

      auto &cliqueSet = clique[neighbor];
      cliqueSet.insert(liveRange);
      pq.push({cliqueSet.size(), neighbor}); // update degree in priority queue

      if (cliqueSet.size() + 1 > maxClique.size()) {
        maxClique = cliqueSet;
        maxClique.insert(neighbor);
      }
    }
  }
}

void MaximumCardinalitySearch::dump(llvm::raw_ostream &os) const {
  os << "Maximum Cardinality Search dump:\n";

  os << "SEO: ";
  for (auto I = simplicialElimOrder.begin(), E = simplicialElimOrder.end();
       I != E; ++I) {
    if (I != simplicialElimOrder.begin())
      os << ", ";
    os << "L" << liveRangeIdMap.at(*I);
  }
  os << "\n";

  os << "Max Clique: ";
  llvm::SmallVector<LiveRange> sortedClique(maxClique.begin(), maxClique.end());
  llvm::sort(sortedClique, [&](LiveRange a, LiveRange b) {
    return liveRangeIdMap.at(a) < liveRangeIdMap.at(b);
  });
  for (auto I = sortedClique.begin(), E = sortedClique.end(); I != E; ++I) {
    if (I != sortedClique.begin())
      os << ", ";
    os << "L" << liveRangeIdMap.at(*I);
  }
  os << "\n";
}

GraphColoring::Color GraphColoring::getColor(LiveRange liveRange) const {
  return colorMap.at(liveRange);
}

void GraphColoring::coloring(InterferenceGraph *intefGraph) {
  MaximumCardinalitySearch *mcs = intefGraph->getMCS();
  const auto &seo = mcs->getSimplicialElimOrder();

  for (LiveRange lr : seo) {
    Color color = findAvailableColor(lr);
    colorMap[lr] = color;
    for (LiveRange neighbor : intefGraph->graph[lr]) {
      neighborColors[neighbor].insert(color);
    }
  }

  auto currLRIdMap = liveRangeAnalysis->getCurrLRIdMap();
  for (const auto &[liveRange, _] : currLRIdMap) {
    if (!colorMap.contains(liveRange))
      colorMap[liveRange] = 0;
  }
}

GraphColoring::Color GraphColoring::createNewColor() { return numColors++; }

GraphColoring::Color GraphColoring::findAvailableColor(LiveRange liveRange) {
  assert(colorMap.find(liveRange) == colorMap.end() &&
         "Live range already colored");

  auto &neighborColorSet = neighborColors[liveRange];
  Color color = 0;
  for (; neighborColorSet.contains(color); ++color)
    ;

  assert(color <= numColors &&
         "Color exceeds the number of colors used so far");
  if (color == numColors)
    numColors++;

  return color;
}

} // namespace kecc
