#include "kecc/ir/IRAttributes.h"
#include "kecc/ir/IRInstructions.h"
#include "kecc/ir/IRTransforms.h"
#include "kecc/ir/PatternMatch.h"
#include "kecc/utils/LogicalResult.h"
#include "llvm/ADT/SmallVectorExtras.h"

namespace kecc::ir {

class InlineCallPattern : public InstPattern<inst::Call> {
public:
  InlineCallPattern() {}

  utils::LogicalResult matchAndRewrite(IRRewriter &rewriter,
                                       inst::Call call) override {
    auto func = call.getFunction();
    if (auto constant = func.getDefiningInst<inst::Constant>()) {
      auto variable = constant.getValue().cast<ConstantVariableAttr>();
      auto name = variable.getName();
      auto funcT =
          func.getType().cast<PointerT>().getPointeeType().cast<FunctionT>();

      auto inlineCall = rewriter.create<inst::InlineCall>(
          call.getRange(), name, funcT,
          llvm::map_to_vector(call.getArguments(),
                              [](Value arg) { return arg; }));
      rewriter.replaceInst(call.getStorage(), inlineCall->getResults());
      return utils::LogicalResult::success();
    } else
      return utils::LogicalResult::failure();
  }
};

PassResult InlineCallPass::run(Module *module) {
  PatternSet set;
  set.addPatterns<InlineCallPattern>();

  auto result = applyPatternConversion(module, set);
  assert(result.succeeded());
  return PassResult::success();
}

} // namespace kecc::ir
