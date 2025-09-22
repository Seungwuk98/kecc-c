#include "kecc/translate/TranslatePasses.h"
#include "kecc/ir/IRTransforms.h"

namespace kecc::translate {

void addO1Passes(ir::PassManager &pm) {
  pm.addPass<ir::CanonicalizeConstant>();
  pm.addPass<ir::InstructionFold>();
  pm.addPass<ir::SimplifyCFG>();
  pm.addPass<ir::Mem2Reg>();
  pm.addPass<ir::DeadCode>();
  pm.addPass<ir::GVN>();
  pm.addPass<InlineMemoryInstPass>();
}

} // namespace kecc::translate
