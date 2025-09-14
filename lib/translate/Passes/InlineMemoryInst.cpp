#include "kecc/ir/IRAttributes.h"
#include "kecc/ir/IRInstructions.h"
#include "kecc/ir/PatternMatch.h"
#include "kecc/translate/IntermediateInsts.h"
#include "kecc/translate/TranslateConstants.h"
#include "kecc/translate/TranslatePasses.h"

namespace kecc::translate {

void InlineMemoryInstPass::init(ir::Module *module) {
  if (!module->getContext()->isRegisteredInst<inst::LoadOffset>()) {
    module->getContext()->registerInst<inst::LoadOffset>();
  }
  if (!module->getContext()->isRegisteredInst<inst::StoreOffset>()) {
    module->getContext()->registerInst<inst::StoreOffset>();
  }
}

class StoreInlinePattern : public ir::InstConversionPattern<ir::inst::Store> {
public:
  StoreInlinePattern() = default;

  utils::LogicalResult matchAndRewrite(ir::IRRewriter &rewriter,
                                       Adaptor adaptor, ir::inst::Store store) {
    auto ptr = adaptor.getPointer();
    if (auto gep = ptr.getDefiningInst<ir::inst::Gep>()) {
      auto offset = gep.getOffset();
      if (!offset.isConstant())
        return utils::LogicalResult::failure();

      auto offsetValue =
          static_cast<std::int64_t>(offset.getDefiningInst<ir::inst::Constant>()
                                        .getValue()
                                        .cast<ir::ConstantIntAttr>()
                                        .getValue());
      if (MIN_INT_12 <= offsetValue && offsetValue <= MAX_INT_12) {
        auto newStore = rewriter.create<inst::StoreOffset>(
            store.getRange(), adaptor.getValue(), gep.getBasePointer(),
            offsetValue);
        rewriter.removeInst(store.getStorage());
        return utils::LogicalResult::success();
      }
    }
    return utils::LogicalResult::failure();
  }
};

class LoadInlinePattern : public ir::InstConversionPattern<ir::inst::Load> {
public:
  LoadInlinePattern() = default;

  utils::LogicalResult matchAndRewrite(ir::IRRewriter &rewriter,
                                       Adaptor adaptor, ir::inst::Load load) {
    auto ptr = adaptor.getPointer();
    if (auto gep = ptr.getDefiningInst<ir::inst::Gep>()) {
      auto offset = gep.getOffset();
      if (!offset.isConstant())
        return utils::LogicalResult::failure();

      auto offsetValue =
          static_cast<std::int64_t>(offset.getDefiningInst<ir::inst::Constant>()
                                        .getValue()
                                        .cast<ir::ConstantIntAttr>()
                                        .getValue());
      if (MIN_INT_12 <= offsetValue && offsetValue <= MAX_INT_12) {
        auto newLoad = rewriter.create<inst::LoadOffset>(
            load.getRange(), gep.getBasePointer(), offsetValue, load.getType());
        rewriter.replaceInst(load.getStorage(), newLoad->getResults());
        return utils::LogicalResult::success();
      }
    }
    return utils::LogicalResult::failure();
  }
};

class GepInlinePattern : public ir::InstConversionPattern<ir::inst::Gep> {
public:
  GepInlinePattern() = default;

  utils::LogicalResult matchAndRewrite(ir::IRRewriter &rewriter,
                                       Adaptor adaptor, ir::inst::Gep gep) {
    auto basePtr = adaptor.getBasePointer();
    auto offset = adaptor.getOffset();
    if (!offset.isConstant())
      return utils::LogicalResult::failure();

    if (auto prevGep = basePtr.getDefiningInst<ir::inst::Gep>()) {
      auto prevBasePtr = prevGep.getBasePointer();
      auto prevOffset = prevGep.getOffset();
      if (!prevOffset.isConstant())
        return utils::LogicalResult::failure();

      auto prevOffsetValue = static_cast<std::int64_t>(
          prevOffset.getDefiningInst<ir::inst::Constant>()
              .getValue()
              .cast<ir::ConstantIntAttr>()
              .getValue());
      auto offsetValue =
          static_cast<std::int64_t>(offset.getDefiningInst<ir::inst::Constant>()
                                        .getValue()
                                        .cast<ir::ConstantIntAttr>()
                                        .getValue());
      auto newOffsetValue = prevOffsetValue + offsetValue;
      ir::Value newOffset;
      {
        ir::IRRewriter::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(
            gep.getParentBlock()->getParentIR()->getConstantBlock());
        newOffset = rewriter.create<ir::inst::Constant>(
            {}, ir::ConstantIntAttr::get(rewriter.getContext(), newOffsetValue,
                                         64, true));
      }
      auto newGep = rewriter.create<ir::inst::Gep>(gep.getRange(), prevBasePtr,
                                                   newOffset, gep.getType());
      rewriter.replaceInst(gep.getStorage(), newGep->getResults());
      return utils::LogicalResult::success();
    }
    return utils::LogicalResult::failure();
  }
};

ir::PassResult InlineMemoryInstPass::run(ir::Module *module) {
  ir::PatternSet patternSet;
  patternSet
      .addPatterns<StoreInlinePattern, LoadInlinePattern, GepInlinePattern>();

  auto result = ir::applyPatternConversion(module, patternSet);
  if (!result.succeeded())
    return ir::PassResult::failure();

  return ir::PassResult::success();
}

} // namespace kecc::translate
