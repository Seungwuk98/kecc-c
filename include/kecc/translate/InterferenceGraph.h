#ifndef KECC_TRANSLATE_INTERFERENCE_GRAPH_H
#define KECC_TRANSLATE_INTERFERENCE_GRAPH_H

#include "kecc/ir/IR.h"
#include "kecc/ir/Module.h"
#include "kecc/translate/LiveRange.h"
#include "llvm/ADT/SetVector.h"

namespace kecc {

class MaximumCardinalitySearch;

class InterferenceGraph {
public:
  static std::unique_ptr<InterferenceGraph>
  create(ir::Module *module, ir::Function *func, bool isFloat);

  size_t getDegree(LiveRange liveRange) const;

  // update the interference graph after spilling
  // LiveRangeAnalysis and LivenessAnalysis must be up to date
  void update();

  bool isForFloatType() const { return isFloat; }

  void dump(llvm::raw_ostream &os) const;

private:
  friend class MaximumCardinalitySearch;

  InterferenceGraph(ir::Module *module, ir::Function *function, bool isFloat,
                    llvm::DenseMap<LiveRange, llvm::DenseSet<LiveRange>> graph)
      : module(module), function(function), isFloat(isFloat),
        graph(std::move(graph)) {}

  ir::Module *module;
  ir::Function *function;
  bool isFloat;
  llvm::DenseMap<LiveRange, llvm::DenseSet<LiveRange>> graph;
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

private:
  void init();
  void search();
  InterferenceGraph *interferenceGraph;

  llvm::SetVector<LiveRange> simplicialElimOrder;
  llvm::DenseSet<LiveRange> maxClique;
  llvm::DenseMap<LiveRange, llvm::DenseSet<LiveRange>> clique;

  llvm::DenseMap<LiveRange, size_t> liveRangeIdMap;
};

} // namespace kecc

#endif // KECC_TRANSLATE_INTERFERENCE_GRAPH_H
