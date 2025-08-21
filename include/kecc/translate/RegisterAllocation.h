#ifndef KECC_TRANSLATE_REGISTER_ALLOCATION_H
#define KECC_TRANSLATE_REGISTER_ALLOCATION_H

#include "kecc/asm/Register.h"
#include "kecc/ir/Module.h"
#include "kecc/translate/InterferenceGraph.h"
#include "kecc/translate/LiveRange.h"
#include "kecc/translate/LiveRangeAnalyses.h"
#include "kecc/translate/SpillAnalysis.h"
#include "kecc/translate/TranslateContext.h"

namespace kecc {

class RegisterAllocation {
public:
  RegisterAllocation(ir::Module *module, TranslateContext *translateContext);
  using Color = GraphColoring::Color;

  as::Register getRegister(ir::Value value);

private:
  void allocateRegisters();

  void allocateRegistersForFunction(ir::Function *function,
                                    InterferenceGraph *intInterferenceGraph,
                                    InterferenceGraph *floatInterferenceGraph);

  ir::Module *module;
  TranslateContext *translateContext;
  SpillAnalysis *spillAnalysis;
  LiveRangeAnalysis *liveRangeAnalysis;

  llvm::SmallVector<as::Register, 8> intArgRegisters;
  llvm::SmallVector<as::Register, 8> floatArgRegisters;

  llvm::DenseMap<
      ir::Function *,
      llvm::DenseMap<std::pair<as::RegisterType, Color>, as::Register>>
      colorToRegisterMap;

  llvm::DenseMap<ir::Function *, llvm::DenseMap<LiveRange, as::Register>>
      liveRangeToRegisterMap;
};

} // namespace kecc

#endif // KECC_TRANSLATE_REGISTER_ALLOCATION_H
