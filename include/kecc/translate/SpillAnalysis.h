#ifndef KECC_TRANSLATE_SPILL_ANALYSIS_H
#define KECC_TRANSLATE_SPILL_ANALYSIS_H

#include "kecc/asm/Register.h"
#include "kecc/ir/IR.h"
#include "kecc/ir/IRAnalyses.h"
#include "kecc/translate/InterferenceGraph.h"
#include "kecc/translate/LiveRange.h"
#include "kecc/translate/LiveRangeAnalyses.h"
#include "kecc/translate/TranslateContext.h"

namespace kecc {

static constexpr long double SPILL_COST_BASE = 10;

class SpillAnalysis : public ir::Analysis {
public:
  // try spill
  // returns true if some values were spilled
  // if iter is 0, it will try to spill until no more values can be spilled
  bool trySpill(size_t iter = 0);

  bool trySpill(ir::Function *function);

  bool trySpill(ir::Function *function, as::RegisterType regType);

  void spillFull();

  const SpillInfo &getSpillInfo() const { return spillInfo; }

  InterferenceGraph *getInterferenceGraph(ir::Function *function,
                                          as::RegisterType regType) const {
    const auto &[forInt, forFloat] = interfGraphMap.at(function);
    return (regType == as::RegisterType::Integer) ? forInt.get()
                                                  : forFloat.get();
  }

  static std::unique_ptr<SpillAnalysis>
  create(ir::Module *module, TranslateContext *translateContext);

  void dumpInterferenceGraph(llvm::raw_ostream &os) const;

private:
  SpillAnalysis(ir::Module *module, TranslateContext *translateContext,
                llvm::DenseMap<ir::Function *,
                               std::pair<std::unique_ptr<InterferenceGraph>,
                                         std::unique_ptr<InterferenceGraph>>>
                    interfGraph)
      : ir::Analysis(module), translateContext(translateContext),
        interfGraphMap(std::move(interfGraph)), spillInfo() {}

  TranslateContext *translateContext;
  llvm::DenseMap<ir::Function *, std::pair<std::unique_ptr<InterferenceGraph>,
                                           std::unique_ptr<InterferenceGraph>>>
      interfGraphMap;
  SpillInfo spillInfo;
};

class SpillCost {
public:
  SpillCost(ir::Module *module, ir::Function *function,
            InterferenceGraph *interfGraph)
      : module(module), function(function), interfGraph(interfGraph) {
    estimateSpillCost();
  }

  long double getSpillCost(LiveRange liveRange) const;

  void dump(llvm::raw_ostream &os) const;

private:
  void estimateSpillCost();

  ir::Module *module;
  ir::Function *function;
  InterferenceGraph *interfGraph;
  llvm::DenseMap<LiveRange, long double> spillCostMap;
};

} // namespace kecc

#endif // KECC_TRANSLATE_SPILL_ANALYSIS_H
