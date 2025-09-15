#include "kecc/ir/IRAttributes.h"
#include "kecc/ir/IRInstructions.h"
#include "kecc/ir/IRTypes.h"
#include "kecc/ir/PatternMatch.h"
#include "kecc/translate/IntermediateInsts.h"
#include "kecc/translate/TranslatePasses.h"
#include "kecc/utils/LogicalResult.h"

namespace kecc::translate {

ir::Instruction createCopy(ir::IRRewriter &rewriter, llvm::SMRange range,
                           ir::Value value, ir::Type retType) {
  if (value.isConstant()) {
    // create outline constant
    return rewriter.create<ir::inst::OutlineConstant>(range, value);
  }

  return rewriter.create<translate::inst::Copy>(range, value, retType);
}

class BinaryCopyPattern : public ir::InstPattern<ir::inst::Binary> {
public:
  BinaryCopyPattern() {};

  utils::LogicalResult matchAdd(ir::IRRewriter &rewriter,
                                ir::inst::Binary binary) {
    auto lhs = binary.getLhs();
    auto rhs = binary.getRhs();
    auto type = binary.getLhs().getType();

    auto rewriteAdd = [&](ir::inst::Constant constant,
                          ir::Value other) -> bool {
      if (type.isa<ir::FloatT>()) {
        auto value = constant.getValue();
        assert(
            value.isa<ir::ConstantFloatAttr>() &&
            "Expected float constant. Use `CanonicalizeConstant` pass first");

        auto floatValue =
            constant.getValue().cast<ir::ConstantFloatAttr>().getValue();
        if (floatValue.isZero()) {
          rewriter.replaceInst(binary.getStorage(), other);
          return true;
        }
      } else if (type.isa<ir::IntT>()) {
        auto value = constant.getValue().cast<ir::ConstantIntAttr>().getValue();
        if (value == 0) {
          rewriter.replaceInst(binary.getStorage(), other);
          return true;
        }
      }
      return false;
    };

    if (lhs.isConstant()) {
      auto constant = lhs.getDefiningInst<ir::inst::Constant>();
      auto replaced = rewriteAdd(constant, rhs);
      if (replaced)
        return utils::LogicalResult::success();
    }

    if (rhs.isConstant()) {
      auto constant = rhs.getDefiningInst<ir::inst::Constant>();
      auto replaced = rewriteAdd(constant, lhs);
      if (replaced)
        return utils::LogicalResult::success();
    }

    return utils::LogicalResult::failure();
  }

  utils::LogicalResult matchSub(ir::IRRewriter &rewriter,
                                ir::inst::Binary binary) {
    auto lhs = binary.getLhs();
    auto rhs = binary.getRhs();
    auto type = binary.getLhs().getType();

    if (rhs.isConstant()) {
      auto constant = rhs.getDefiningInst<ir::inst::Constant>();
      if (type.isa<ir::FloatT>()) {
        auto value =
            constant.getValue().cast<ir::ConstantFloatAttr>().getValue();
        if (value.isZero()) {
          rewriter.replaceInst(binary.getStorage(), lhs);
          return utils::LogicalResult::success();
        }
      } else if (type.isa<ir::IntT>()) {
        auto value = constant.getValue().cast<ir::ConstantIntAttr>().getValue();
        if (value == 0) {
          rewriter.replaceInst(binary.getStorage(), lhs);
          return utils::LogicalResult::success();
        }
      }
    }
    return utils::LogicalResult::failure();
  }

  utils::LogicalResult matchMul(ir::IRRewriter &rewriter,
                                ir::inst::Binary binary) {
    auto lhs = binary.getLhs();
    auto rhs = binary.getRhs();
    auto type = binary.getLhs().getType();

    auto rewriteMul = [&](ir::inst::Constant constant,
                          ir::Value other) -> bool {
      if (type.isa<ir::FloatT>()) {
        auto value =
            constant.getValue().cast<ir::ConstantFloatAttr>().getValue();
        if (value.isExactlyValue(1.0)) {
          rewriter.replaceInst(binary.getStorage(), other);
          return true;
        }
      } else if (type.isa<ir::IntT>()) {
        auto value = constant.getValue().cast<ir::ConstantIntAttr>().getValue();
        if (value == 1) {
          rewriter.replaceInst(binary.getStorage(), other);
          return true;
        }
      }
      return false;
    };

    if (lhs.isConstant()) {
      auto constant = lhs.getDefiningInst<ir::inst::Constant>();
      auto replaced = rewriteMul(constant, rhs);
      if (replaced)
        return utils::LogicalResult::success();
    }

    if (rhs.isConstant()) {
      auto constant = rhs.getDefiningInst<ir::inst::Constant>();
      auto replaced = rewriteMul(constant, lhs);
      if (replaced)
        return utils::LogicalResult::success();
    }
    return utils::LogicalResult::failure();
  }

  utils::LogicalResult matchDiv(ir::IRRewriter &rewriter,
                                ir::inst::Binary binary) {
    auto lhs = binary.getLhs();
    auto rhs = binary.getRhs();
    auto type = binary.getLhs().getType();

    if (rhs.isConstant()) {
      auto constant = rhs.getDefiningInst<ir::inst::Constant>();
      if (type.isa<ir::FloatT>()) {
        auto value =
            constant.getValue().cast<ir::ConstantFloatAttr>().getValue();
        if (value.isExactlyValue(1.0)) {
          rewriter.replaceInst(binary.getStorage(), lhs);
          return utils::LogicalResult::success();
        }
      } else if (type.isa<ir::IntT>()) {
        auto value = constant.getValue().cast<ir::ConstantIntAttr>().getValue();
        if (value == 1) {
          rewriter.replaceInst(binary.getStorage(), lhs);
          return utils::LogicalResult::success();
        }
      }
    }
    return utils::LogicalResult::failure();
  }

  utils::LogicalResult matchAnd(ir::IRRewriter &rewriter,
                                ir::inst::Binary binary) {
    auto lhs = binary.getLhs();
    auto rhs = binary.getRhs();
    auto type = binary.getLhs().getType();

    auto rewriteAnd = [&](ir::inst::Constant constant,
                          ir::Value other) -> bool {
      if (type.isa<ir::IntT>()) {
        auto value = constant.getValue().cast<ir::ConstantIntAttr>();
        auto allOnes = ir::ConstantIntAttr::get(
            type.getContext(), -1, value.getBitWidth(), value.isSigned());
        if (value == allOnes) {
          rewriter.replaceInst(binary.getStorage(), other);
          return true;
        }
      }
      return false;
    };

    if (lhs.isConstant()) {
      auto constant = lhs.getDefiningInst<ir::inst::Constant>();
      auto replaced = rewriteAnd(constant, rhs);
      if (replaced)
        return utils::LogicalResult::success();
    }

    if (rhs.isConstant()) {
      auto constant = rhs.getDefiningInst<ir::inst::Constant>();
      auto replaced = rewriteAnd(constant, lhs);
      if (replaced)
        return utils::LogicalResult::success();
    }
    return utils::LogicalResult::failure();
  }

  utils::LogicalResult matchOr(ir::IRRewriter &rewriter,
                               ir::inst::Binary binary) {
    auto lhs = binary.getLhs();
    auto rhs = binary.getRhs();
    auto type = binary.getLhs().getType();

    auto rewriteOr = [&](ir::inst::Constant constant, ir::Value other) -> bool {
      if (type.isa<ir::IntT>()) {
        auto value = constant.getValue().cast<ir::ConstantIntAttr>().getValue();
        if (value == 0) {
          rewriter.replaceInst(binary.getStorage(), other);
          return true;
        }
      }
      return false;
    };

    if (lhs.isConstant()) {
      auto constant = lhs.getDefiningInst<ir::inst::Constant>();
      auto replaced = rewriteOr(constant, rhs);
      if (replaced)
        return utils::LogicalResult::success();
    }

    if (rhs.isConstant()) {
      auto constant = rhs.getDefiningInst<ir::inst::Constant>();
      auto replaced = rewriteOr(constant, lhs);
      if (replaced)
        return utils::LogicalResult::success();
    }
    return utils::LogicalResult::failure();
  }

  utils::LogicalResult matchXor(ir::IRRewriter &rewriter,
                                ir::inst::Binary binary) {
    auto lhs = binary.getLhs();
    auto rhs = binary.getRhs();
    auto type = binary.getLhs().getType();

    auto rewriteXor = [&](ir::inst::Constant constant,
                          ir::Value other) -> bool {
      if (type.isa<ir::IntT>()) {
        auto value = constant.getValue().cast<ir::ConstantIntAttr>().getValue();
        if (value == 0) {
          rewriter.replaceInst(binary.getStorage(), other);
          return true;
        }
      }
      return false;
    };

    if (lhs.isConstant()) {
      auto constant = lhs.getDefiningInst<ir::inst::Constant>();
      auto replaced = rewriteXor(constant, rhs);
      if (replaced)
        return utils::LogicalResult::success();
    }

    if (rhs.isConstant()) {
      auto constant = rhs.getDefiningInst<ir::inst::Constant>();
      auto replaced = rewriteXor(constant, lhs);
      if (replaced)
        return utils::LogicalResult::success();
    }

    return utils::LogicalResult::failure();
  }

  utils::LogicalResult matchAndRewrite(ir::IRRewriter &rewriter,
                                       ir::inst::Binary binary) override {
    auto lhs = binary.getLhs();
    auto rhs = binary.getRhs();

    bool lhsIsConstant = lhs.isConstant();
    bool rhsIsConstant = rhs.isConstant();

    if (!lhsIsConstant && !rhsIsConstant)
      return utils::LogicalResult::failure();

    switch (binary.getOpKind()) {
    case ir::inst::Binary::OpKind::Add:
      return matchAdd(rewriter, binary);
    case ir::inst::Binary::OpKind::Sub:
      return matchSub(rewriter, binary);
    case ir::inst::Binary::OpKind::Mul:
      return matchMul(rewriter, binary);
    case ir::inst::Binary::OpKind::Div:
      return matchDiv(rewriter, binary);
    case ir::inst::Binary::OpKind::BitAnd:
      return matchAnd(rewriter, binary);
    case ir::inst::Binary::OpKind::BitOr:
      return matchOr(rewriter, binary);
    case ir::inst::Binary::OpKind::BitXor:
      return matchXor(rewriter, binary);
    default:
      return utils::LogicalResult::failure();
    }
  }
};

class TypeCastCopyPattern : public ir::InstPattern<ir::inst::TypeCast> {
public:
  TypeCastCopyPattern() {};

  utils::LogicalResult matchAndRewrite(ir::IRRewriter &rewriter,
                                       ir::inst::TypeCast typeCast) override {
    auto target = typeCast.getValue();
    if (target.getType() == typeCast.getType()) {
      rewriter.replaceInst(typeCast.getStorage(), target);
      return utils::LogicalResult::success();
    }

    if (target.getType().isa<ir::PointerT>() &&
        typeCast.getType().isa<ir::PointerT>()) {
      auto copy = createCopy(rewriter, typeCast->getRange(), target,
                             typeCast.getType());
      rewriter.replaceInst(typeCast.getStorage(), copy->getResults());
      return utils::LogicalResult::success();
    }
    return utils::LogicalResult::failure();
  }
};

class GepCopyPattern : public ir::InstPattern<ir::inst::Gep> {
public:
  GepCopyPattern() {};

  utils::LogicalResult matchAndRewrite(ir::IRRewriter &rewriter,
                                       ir::inst::Gep gep) override {
    auto offset = gep.getOffset();
    if (offset.isConstant()) {
      auto intValue = offset.getDefiningInst<ir::inst::Constant>()
                          .getValue()
                          .cast<ir::ConstantIntAttr>()
                          .getValue();
      if (!intValue) {
        if (gep.getType() == gep.getBasePointer().getType()) {
          rewriter.replaceInst(gep.getStorage(), gep.getBasePointer());
        } else {
          auto copy = createCopy(rewriter, gep->getRange(),
                                 gep.getBasePointer(), gep.getType());
          rewriter.replaceInst(gep.getStorage(), copy->getResults());
        }
        return utils::LogicalResult::success();
      }
    }

    return utils::LogicalResult::failure();
  }
};

class UnaryCopyPattern : public ir::InstPattern<ir::inst::Unary> {
public:
  UnaryCopyPattern() {};

  utils::LogicalResult matchAndRewrite(ir::IRRewriter &rewriter,
                                       ir::inst::Unary unary) override {
    if (unary.getOpKind() == ir::inst::Unary::Plus) {
      rewriter.replaceInst(unary.getStorage(), unary.getValue());
      return utils::LogicalResult::success();
    }

    return utils::LogicalResult::failure();
  }
};

void ConversionToCopyPass::init(ir::Module *module) {
  if (!module->getContext()->isRegisteredInst<translate::inst::Copy>()) {
    module->getContext()->registerInst<translate::inst::Copy>();
  }

  if (!module->getContext()->isRegisteredInst<ir::inst::OutlineConstant>()) {
    module->getContext()->registerInst<ir::inst::OutlineConstant>();
  }
}

ir::PassResult ConversionToCopyPass::run(ir::Module *module) {
  ir::PatternSet patternSet;
  patternSet.addPatterns<BinaryCopyPattern, UnaryCopyPattern, GepCopyPattern,
                         TypeCastCopyPattern>();

  auto result = applyPatternConversion(module, patternSet);
  if (!result.succeeded())
    return ir::PassResult::failure();

  return ir::PassResult::success();
}

} // namespace kecc::translate
