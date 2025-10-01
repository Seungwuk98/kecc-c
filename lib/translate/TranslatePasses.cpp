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

void registerDefaultTranslationPasses(ir::PassManager &pm) {
  pm.addPass<ir::CFGReach>();
  pm.addPass<ir::CanonicalizeStruct>();
  pm.addPass<ir::FoldTypeCast>();
  pm.addPass<ir::InlineCallPass>();
  pm.addPass<ir::CanonicalizeConstant>();
  pm.addPass<translate::ConversionToCopyPass>();
  pm.addPass<ir::OutlineConstantPass>();
  pm.addPass<ir::OutlineMultipleResults>();
  pm.addPass<ir::CreateFunctionArgument>();
  pm.addPass<ir::CanonicalizeConstant>();
}

} // namespace kecc::translate
