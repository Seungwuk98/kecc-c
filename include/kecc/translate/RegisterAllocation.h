#ifndef KECC_TRANSLATE_REGISTER_ALLOCATION_H
#define KECC_TRANSLATE_REGISTER_ALLOCATION_H

#include "kecc/asm/Register.h"
#include "kecc/ir/IRAnalyses.h"
#include "kecc/ir/Module.h"
#include "kecc/translate/InterferenceGraph.h"
#include "kecc/translate/LiveRange.h"
#include "kecc/translate/LiveRangeAnalyses.h"
#include "kecc/translate/SpillAnalysis.h"

namespace kecc {

class RegisterAllocation {
public:
  as::Register getRegister(ir::Value value) const;
  as::Register getRegister(LiveRange liveRange) const;

private:
  ir::Module *module;
  SpillAnalysis *spillAnalysis;
  LiveRangeAnalysis *liveRangeAnalysis;
  InterferenceGraph *interferenceGraph;
};

} // namespace kecc

#endif // KECC_TRANSLATE_REGISTER_ALLOCATION_H
