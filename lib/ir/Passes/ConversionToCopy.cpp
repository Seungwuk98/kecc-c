#include "kecc/ir/IRAttributes.h"
#include "kecc/ir/IRInstructions.h"
#include "kecc/ir/IRTransforms.h"
#include "kecc/ir/IRTypes.h"
#include "kecc/ir/PatternMatch.h"
#include "kecc/utils/LogicalResult.h"

namespace kecc::ir {

Instruction createCopy(IRRewriter &rewriter, llvm::SMRange range, Value value,
                       Type retType) {
  if (value.isConstant()) {
    // create outline constant
    return rewriter.create<inst::OutlineConstant>(range, value);
  }

  return rewriter.create<inst::Copy>(range, value, retType);
}

class BinaryCopyPattern : public InstPattern<inst::Binary> {
public:
  BinaryCopyPattern() {};

  utils::LogicalResult matchAdd(IRRewriter &rewriter, inst::Binary binary) {
    auto lhs = binary.getLhs();
    auto rhs = binary.getRhs();
    auto type = binary.getLhs().getType();

    auto rewriteAdd = [&](inst::Constant constant, Value other) -> bool {
      if (type.isa<FloatT>()) {
        auto value = constant.getValue().cast<ConstantFloatAttr>().getValue();
        if (value.isZero()) {
          rewriter.replaceInst(binary.getStorage(), other);
          return true;
        }
      } else if (type.isa<IntT>()) {
        auto value = constant.getValue().cast<ConstantIntAttr>().getValue();
        if (value == 0) {
          rewriter.replaceInst(binary.getStorage(), other);
          return true;
        }
      }
      return false;
    };

    if (lhs.isConstant()) {
      auto constant = lhs.getDefiningInst<inst::Constant>();
      auto replaced = rewriteAdd(constant, rhs);
      if (replaced)
        return utils::LogicalResult::success();
    }

    if (rhs.isConstant()) {
      auto constant = rhs.getDefiningInst<inst::Constant>();
      auto replaced = rewriteAdd(constant, lhs);
      if (replaced)
        return utils::LogicalResult::success();
    }

    return utils::LogicalResult::failure();
  }

  utils::LogicalResult matchSub(IRRewriter &rewriter, inst::Binary binary) {
    auto lhs = binary.getLhs();
    auto rhs = binary.getRhs();
    auto type = binary.getLhs().getType();

    if (rhs.isConstant()) {
      auto constant = rhs.getDefiningInst<inst::Constant>();
      if (type.isa<FloatT>()) {
        auto value = constant.getValue().cast<ConstantFloatAttr>().getValue();
        if (value.isZero()) {
          rewriter.replaceInst(binary.getStorage(), lhs);
          return utils::LogicalResult::success();
        }
      } else if (type.isa<IntT>()) {
        auto value = constant.getValue().cast<ConstantIntAttr>().getValue();
        if (value == 0) {
          rewriter.replaceInst(binary.getStorage(), lhs);
          return utils::LogicalResult::success();
        }
      }
    }
    return utils::LogicalResult::failure();
  }

  utils::LogicalResult matchMul(IRRewriter &rewriter, inst::Binary binary) {
    auto lhs = binary.getLhs();
    auto rhs = binary.getRhs();
    auto type = binary.getLhs().getType();

    auto rewriteMul = [&](inst::Constant constant, Value other) -> bool {
      if (type.isa<FloatT>()) {
        auto value = constant.getValue().cast<ConstantFloatAttr>().getValue();
        if (value.isExactlyValue(1.0)) {
          rewriter.replaceInst(binary.getStorage(), other);
          return true;
        }
      } else if (type.isa<IntT>()) {
        auto value = constant.getValue().cast<ConstantIntAttr>().getValue();
        if (value == 1) {
          rewriter.replaceInst(binary.getStorage(), other);
          return true;
        }
      }
      return false;
    };

    if (lhs.isConstant()) {
      auto constant = lhs.getDefiningInst<inst::Constant>();
      auto replaced = rewriteMul(constant, rhs);
      if (replaced)
        return utils::LogicalResult::success();
    }

    if (rhs.isConstant()) {
      auto constant = rhs.getDefiningInst<inst::Constant>();
      auto replaced = rewriteMul(constant, lhs);
      if (replaced)
        return utils::LogicalResult::success();
    }
    return utils::LogicalResult::failure();
  }

  utils::LogicalResult matchDiv(IRRewriter &rewriter, inst::Binary binary) {
    auto lhs = binary.getLhs();
    auto rhs = binary.getRhs();
    auto type = binary.getLhs().getType();

    if (rhs.isConstant()) {
      auto constant = rhs.getDefiningInst<inst::Constant>();
      if (type.isa<FloatT>()) {
        auto value = constant.getValue().cast<ConstantFloatAttr>().getValue();
        if (value.isExactlyValue(1.0)) {
          rewriter.replaceInst(binary.getStorage(), lhs);
          return utils::LogicalResult::success();
        }
      } else if (type.isa<IntT>()) {
        auto value = constant.getValue().cast<ConstantIntAttr>().getValue();
        if (value == 1) {
          rewriter.replaceInst(binary.getStorage(), lhs);
          return utils::LogicalResult::success();
        }
      }
    }
    return utils::LogicalResult::failure();
  }

  utils::LogicalResult matchAnd(IRRewriter &rewriter, inst::Binary binary) {
    auto lhs = binary.getLhs();
    auto rhs = binary.getRhs();
    auto type = binary.getLhs().getType();

    auto rewriteAnd = [&](inst::Constant constant, Value other) -> bool {
      if (type.isa<IntT>()) {
        auto value = constant.getValue().cast<ConstantIntAttr>();
        auto allOnes = ConstantIntAttr::get(
            type.getContext(), -1, value.getBitWidth(), value.isSigned());
        if (value == allOnes) {
          rewriter.replaceInst(binary.getStorage(), other);
          return true;
        }
      }
      return false;
    };

    if (lhs.isConstant()) {
      auto constant = lhs.getDefiningInst<inst::Constant>();
      auto replaced = rewriteAnd(constant, rhs);
      if (replaced)
        return utils::LogicalResult::success();
    }

    if (rhs.isConstant()) {
      auto constant = rhs.getDefiningInst<inst::Constant>();
      auto replaced = rewriteAnd(constant, lhs);
      if (replaced)
        return utils::LogicalResult::success();
    }
    return utils::LogicalResult::failure();
  }

  utils::LogicalResult matchOr(IRRewriter &rewriter, inst::Binary binary) {
    auto lhs = binary.getLhs();
    auto rhs = binary.getRhs();
    auto type = binary.getLhs().getType();

    auto rewriteOr = [&](inst::Constant constant, Value other) -> bool {
      if (type.isa<IntT>()) {
        auto value = constant.getValue().cast<ConstantIntAttr>().getValue();
        if (value == 0) {
          rewriter.replaceInst(binary.getStorage(), other);
          return true;
        }
      }
      return false;
    };

    if (lhs.isConstant()) {
      auto constant = lhs.getDefiningInst<inst::Constant>();
      auto replaced = rewriteOr(constant, rhs);
      if (replaced)
        return utils::LogicalResult::success();
    }

    if (rhs.isConstant()) {
      auto constant = rhs.getDefiningInst<inst::Constant>();
      auto replaced = rewriteOr(constant, lhs);
      if (replaced)
        return utils::LogicalResult::success();
    }
    return utils::LogicalResult::failure();
  }

  utils::LogicalResult matchXor(IRRewriter &rewriter, inst::Binary binary) {
    auto lhs = binary.getLhs();
    auto rhs = binary.getRhs();
    auto type = binary.getLhs().getType();

    auto rewriteXor = [&](inst::Constant constant, Value other) -> bool {
      if (type.isa<IntT>()) {
        auto value = constant.getValue().cast<ConstantIntAttr>().getValue();
        if (value == 0) {
          rewriter.replaceInst(binary.getStorage(), other);
          return true;
        }
      }
      return false;
    };

    if (lhs.isConstant()) {
      auto constant = lhs.getDefiningInst<inst::Constant>();
      auto replaced = rewriteXor(constant, rhs);
      if (replaced)
        return utils::LogicalResult::success();
    }

    if (rhs.isConstant()) {
      auto constant = rhs.getDefiningInst<inst::Constant>();
      auto replaced = rewriteXor(constant, lhs);
      if (replaced)
        return utils::LogicalResult::success();
    }

    return utils::LogicalResult::failure();
  }

  utils::LogicalResult matchAndRewrite(IRRewriter &rewriter,
                                       inst::Binary binary) override {
    auto lhs = binary.getLhs();
    auto rhs = binary.getRhs();

    bool lhsIsConstant = lhs.isConstant();
    bool rhsIsConstant = rhs.isConstant();

    if (!lhsIsConstant && !rhsIsConstant)
      return utils::LogicalResult::failure();

    switch (binary.getOpKind()) {
    case inst::Binary::OpKind::Add:
      return matchAdd(rewriter, binary);
    case inst::Binary::OpKind::Sub:
      return matchSub(rewriter, binary);
    case inst::Binary::OpKind::Mul:
      return matchMul(rewriter, binary);
    case inst::Binary::OpKind::Div:
      return matchDiv(rewriter, binary);
    case inst::Binary::OpKind::BitAnd:
      return matchAnd(rewriter, binary);
    case inst::Binary::OpKind::BitOr:
      return matchOr(rewriter, binary);
    case inst::Binary::OpKind::BitXor:
      return matchXor(rewriter, binary);
    default:
      return utils::LogicalResult::failure();
    }
  }
};

class TypeCastCopyPattern : public InstPattern<inst::TypeCast> {
public:
  TypeCastCopyPattern() {};

  utils::LogicalResult matchAndRewrite(IRRewriter &rewriter,
                                       inst::TypeCast typeCast) override {
    auto target = typeCast.getValue();
    if (target.getType() == typeCast.getType()) {
      rewriter.replaceInst(typeCast.getStorage(), target);
      return utils::LogicalResult::success();
    }

    if (target.getType().isa<PointerT>() &&
        typeCast.getType().isa<PointerT>()) {
      auto copy = createCopy(rewriter, typeCast->getRange(), target,
                             typeCast.getType());
      rewriter.replaceInst(typeCast.getStorage(), copy->getResults());
      return utils::LogicalResult::success();
    }
    return utils::LogicalResult::failure();
  }
};

class GepCopyPattern : public InstPattern<inst::Gep> {
public:
  GepCopyPattern() {};

  utils::LogicalResult matchAndRewrite(IRRewriter &rewriter,
                                       inst::Gep gep) override {
    auto offset = gep.getOffset();
    if (offset.isConstant()) {
      auto intValue = offset.getDefiningInst<inst::Constant>()
                          .getValue()
                          .cast<ConstantIntAttr>()
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

class UnaryCopyPattern : public InstPattern<inst::Unary> {
public:
  UnaryCopyPattern() {};

  utils::LogicalResult matchAndRewrite(IRRewriter &rewriter,
                                       inst::Unary unary) override {
    if (unary.getOpKind() == inst::Unary::Plus) {
      rewriter.replaceInst(unary.getStorage(), unary.getValue());
      return utils::LogicalResult::success();
    }

    return utils::LogicalResult::failure();
  }
};

void ConversionToCopyPass::init(Module *module) {
  if (!module->getContext()->isRegisteredInst<inst::Copy>()) {
    module->getContext()->registerInst<inst::Copy>();
  }

  if (!module->getContext()->isRegisteredInst<inst::OutlineConstant>()) {
    module->getContext()->registerInst<inst::OutlineConstant>();
  }
}

PassResult ConversionToCopyPass::run(Module *module) {
  PatternSet patternSet;
  patternSet.addPatterns<BinaryCopyPattern, UnaryCopyPattern, GepCopyPattern,
                         TypeCastCopyPattern>();

  auto result = applyPatternConversion(module, patternSet);
  if (!result.succeeded())
    return PassResult::failure();

  return PassResult::success();
}

} // namespace kecc::ir
