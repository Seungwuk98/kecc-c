#ifndef KECC_TRANSLATE_INTERFERENCE_GRAPH_H
#define KECC_TRANSLATE_INTERFERENCE_GRAPH_H

#include "kecc/ir/IR.h"
#include "kecc/ir/Module.h"
#include "kecc/translate/LiveRange.h"
#include "llvm/ADT/SetVector.h"

namespace kecc {

class MaximumCardinalitySearch;
class LiveRangeAnalysis;
class GraphColoring;

class InterferenceGraph {
public:
  ~InterferenceGraph();

  static std::unique_ptr<InterferenceGraph>
  create(ir::Module *module, ir::Function *func, bool isFloat);

  size_t getDegree(LiveRange liveRange) const;

  // update the interference graph after spilling
  // LiveRangeAnalysis and LivenessAnalysis must be up to date
  void update();

  bool isForFloatType() const { return isFloat; }

  void dump(llvm::raw_ostream &os) const;

  MaximumCardinalitySearch *getMCS() const { return mcs.get(); }
  GraphColoring *getGraphColoring() const { return graphColoring.get(); }

  GraphColoring *coloring();

private:
  friend class MaximumCardinalitySearch;
  friend class GraphColoring;

  InterferenceGraph(ir::Module *module, ir::Function *function, bool isFloat,
                    llvm::DenseMap<LiveRange, llvm::DenseSet<LiveRange>> graph);

  ir::Module *module;
  ir::Function *function;
  bool isFloat;
  llvm::DenseMap<LiveRange, llvm::DenseSet<LiveRange>> graph;
  std::unique_ptr<MaximumCardinalitySearch> mcs;
  std::unique_ptr<GraphColoring> graphColoring;
};

void spill(ir::Module *module, InterferenceGraph *graph);

class MaximumCardinalitySearch {
public:
  MaximumCardinalitySearch(InterferenceGraph *interferenceGraph)
      : interferenceGraph(interferenceGraph) {
    init();
    search();
  }

  MaximumCardinalitySearch(MaximumCardinalitySearch &&) = default;
  MaximumCardinalitySearch &operator=(MaximumCardinalitySearch &&) = default;

  const llvm::DenseSet<LiveRange> &getMaxClique() const { return maxClique; }

  const llvm::SetVector<LiveRange> &getSimplicialElimOrder() const {
    return simplicialElimOrder;
  }

  void dump(llvm::raw_ostream &os) const;

private:
  void init();
  void search();
  InterferenceGraph *interferenceGraph;

  llvm::SetVector<LiveRange> simplicialElimOrder;
  llvm::DenseSet<LiveRange> maxClique;
  llvm::DenseMap<LiveRange, llvm::DenseSet<LiveRange>> clique;

  llvm::DenseMap<LiveRange, size_t> liveRangeIdMap;
};

class GraphColoring {
public:
  using Color = size_t;

  GraphColoring(InterferenceGraph *interfGraph)
      : interferenceGraph(interfGraph),
        liveRangeAnalysis(
            interfGraph->module->getAnalysis<LiveRangeAnalysis>()) {
    assert(liveRangeAnalysis && "LiveRangeAnalysis must not be null");
    initCost(interfGraph->module);
    coloring(interfGraph);
  }

  Color getColor(LiveRange liveRange) const;
  long double getCallerSaveCost(Color color) const;

  size_t getNumColors() const { return numColors; }

private:
  void initCost(ir::Module *module);

  void coloring(InterferenceGraph *interfGraph);

  Color createNewColor();

  Color findAvailableColor(LiveRange liveRange);

  InterferenceGraph *interferenceGraph;
  LiveRangeAnalysis *liveRangeAnalysis;
  llvm::DenseMap<LiveRange, Color> colorMap;
  llvm::DenseMap<LiveRange, long double> saveCostMap;
  llvm::DenseMap<Color, long double> colorSaveCostMap;
  llvm::DenseMap<LiveRange, llvm::DenseSet<Color>> neighborColors;
  Color numColors = 0;
};

} // namespace kecc

#endif // KECC_TRANSLATE_INTERFERENCE_GRAPH_H
