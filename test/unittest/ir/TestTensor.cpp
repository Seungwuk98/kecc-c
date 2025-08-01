#include "TestTensor.h"
#include "kecc/ir/PatternMatch.h"

DEFINE_KECC_TYPE_ID(kecc::ir::TensorT);
DEFINE_KECC_TYPE_ID(kecc::ir::inst::CreateTensor);
DEFINE_KECC_TYPE_ID(kecc::ir::inst::Transpose);

namespace kecc::ir {

class RemoveOverlappedTransposePattern : public InstPattern<inst::Transpose> {
public:
  RemoveOverlappedTransposePattern() : InstPattern() {}

  utils::LogicalResult matchAndRewrite(IRRewriter &rewriter,
                                       inst::Transpose inst) override {
    auto input = inst.getInput();

    if (auto prevTranspose = input.getDefiningInst<inst::Transpose>()) {
      auto prevInput = prevTranspose.getInput();

      rewriter.replaceInst(inst.getStorage(), prevInput);
      return utils::LogicalResult::success(); // success
    }

    return utils::LogicalResult::failure(); // fail
  }
};

void addTransposePattern(PatternSet &pattern) {
  pattern.addPatterns<RemoveOverlappedTransposePattern>();
}

} // namespace kecc::ir
