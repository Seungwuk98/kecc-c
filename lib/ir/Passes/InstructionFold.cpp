#include "kecc/ir/IRAttributes.h"
#include "kecc/ir/IRInstructions.h"
#include "kecc/ir/IRTransforms.h"
#include "kecc/ir/PatternMatch.h"

namespace kecc::ir {

namespace fold {

static ConstantAttr
intOp(ConstantAttr lhs, ConstantAttr rhs,
      llvm::function_ref<ConstantAttr(ConstantIntAttr, ConstantIntAttr)> op) {
  auto lhsInt = lhs.cast<ConstantIntAttr>();
  auto rhsInt = rhs.cast<ConstantIntAttr>();
  return op(lhsInt, rhsInt);
}

static ConstantAttr floatOp(
    ConstantAttr lhs, ConstantAttr rhs,
    llvm::function_ref<ConstantAttr(ConstantFloatAttr, ConstantFloatAttr)> op) {
  auto lhsFloat = lhs.dyn_cast<ConstantFloatAttr>();
  if (!lhsFloat) {
    auto strFloat = lhsFloat.cast<ConstantStringFloatAttr>();
    lhsFloat = strFloat.convertToFloatAttr();
  }
  auto rhsFloat = rhs.dyn_cast<ConstantFloatAttr>();
  if (!rhsFloat) {
    auto strFloat = rhsFloat.cast<ConstantStringFloatAttr>();
    rhsFloat = strFloat.convertToFloatAttr();
  }
  return op(lhsFloat, rhsFloat);
}

static ConstantAttr add(ConstantAttr lhs, ConstantAttr rhs) {
  auto lhsT = lhs.getType();
  auto rhsT = rhs.getType();
  if (lhsT != rhsT)
    return nullptr;

  if (lhsT.isa<IntT>()) {
    return intOp(lhs, rhs, [](ConstantIntAttr lhsInt, ConstantIntAttr rhsInt) {
      return ConstantIntAttr::get(lhsInt.getContext(),
                                  lhsInt.getValue() + rhsInt.getValue(),
                                  lhsInt.getBitWidth(), lhsInt.isSigned());
    });
  } else if (lhsT.isa<FloatT>()) {
    return floatOp(
        lhs, rhs, [](ConstantFloatAttr lhsFloat, ConstantFloatAttr rhsFloat) {
          return ConstantFloatAttr::get(
              lhsFloat.getContext(), lhsFloat.getValue() + rhsFloat.getValue());
        });
  }
  return nullptr;
}

static ConstantAttr sub(ConstantAttr lhs, ConstantAttr rhs) {
  auto lhsT = lhs.getType();
  auto rhsT = rhs.getType();
  if (lhsT != rhsT)
    return nullptr;

  if (lhsT.isa<IntT>()) {
    return intOp(lhs, rhs, [](ConstantIntAttr lhsInt, ConstantIntAttr rhsInt) {
      return ConstantIntAttr::get(lhsInt.getContext(),
                                  lhsInt.getValue() - rhsInt.getValue(),
                                  lhsInt.getBitWidth(), lhsInt.isSigned());
    });
  } else if (lhsT.isa<FloatT>()) {
    return floatOp(
        lhs, rhs, [](ConstantFloatAttr lhsFloat, ConstantFloatAttr rhsFloat) {
          return ConstantFloatAttr::get(
              lhsFloat.getContext(), lhsFloat.getValue() - rhsFloat.getValue());
        });
  }
  return nullptr;
}

static ConstantAttr mul(ConstantAttr lhs, ConstantAttr rhs) {
  auto lhsT = lhs.getType();
  auto rhsT = rhs.getType();
  if (lhsT != rhsT)
    return nullptr;

  if (lhsT.isa<IntT>()) {
    return intOp(lhs, rhs, [](ConstantIntAttr lhsInt, ConstantIntAttr rhsInt) {
      return ConstantIntAttr::get(lhsInt.getContext(),
                                  lhsInt.getValue() * rhsInt.getValue(),
                                  lhsInt.getBitWidth(), lhsInt.isSigned());
    });
  } else if (lhsT.isa<FloatT>()) {
    return floatOp(
        lhs, rhs, [](ConstantFloatAttr lhsFloat, ConstantFloatAttr rhsFloat) {
          return ConstantFloatAttr::get(
              lhsFloat.getContext(), lhsFloat.getValue() * rhsFloat.getValue());
        });
  }
  return nullptr;
}

static ConstantAttr div(ConstantAttr lhs, ConstantAttr rhs) {
  auto lhsT = lhs.getType();
  auto rhsT = rhs.getType();
  if (lhsT != rhsT)
    return nullptr;

  if (lhsT.isa<IntT>()) {
    return intOp(
        lhs, rhs,
        [](ConstantIntAttr lhsInt, ConstantIntAttr rhsInt) -> ConstantIntAttr {
          if (rhsInt.getValue() == 0)
            return nullptr; // Division by zero

          if (lhsInt.isSigned())
            return ConstantIntAttr::get(
                lhsInt.getContext(),
                static_cast<std::int64_t>(lhsInt.getValue()) /
                    static_cast<std::int64_t>(rhsInt.getValue()),
                lhsInt.getBitWidth(), true);
          return ConstantIntAttr::get(lhsInt.getContext(),
                                      lhsInt.getValue() / rhsInt.getValue(),
                                      lhsInt.getBitWidth(), lhsInt.isSigned());
        });
  } else if (lhsT.isa<FloatT>()) {
    return floatOp(lhs, rhs,
                   [](ConstantFloatAttr lhsFloat,
                      ConstantFloatAttr rhsFloat) -> ConstantFloatAttr {
                     if (rhsFloat.getValue().isZero())
                       return ConstantFloatAttr::get(
                           lhsFloat.getContext(),
                           llvm::APFloat::getInf(
                               lhsFloat.getValue()
                                   .getSemantics())); // Division by zero -> inf
                     return ConstantFloatAttr::get(lhsFloat.getContext(),
                                                   lhsFloat.getValue() /
                                                       rhsFloat.getValue());
                   });
  }

  return nullptr;
}

static ConstantAttr mod(ConstantAttr lhs, ConstantAttr rhs) {
  auto lhsT = lhs.getType();
  auto rhsT = rhs.getType();
  if (lhsT != rhsT)
    return nullptr;

  if (lhsT.isa<IntT>()) {
    return intOp(
        lhs, rhs,
        [](ConstantIntAttr lhsInt, ConstantIntAttr rhsInt) -> ConstantIntAttr {
          if (rhsInt.getValue() == 0)
            return nullptr; // Division by zero
          if (lhsInt.isSigned())
            return ConstantIntAttr::get(
                lhsInt.getContext(),
                static_cast<std::int64_t>(lhsInt.getValue()) %
                    static_cast<std::int64_t>(rhsInt.getValue()),
                lhsInt.getBitWidth(), true);
          return ConstantIntAttr::get(lhsInt.getContext(),
                                      lhsInt.getValue() % rhsInt.getValue(),
                                      lhsInt.getBitWidth(), false);
        });
  }
  return nullptr;
}

static ConstantAttr bitAnd(ConstantAttr lhs, ConstantAttr rhs) {
  auto lhsT = lhs.getType();
  auto rhsT = rhs.getType();
  if (lhsT != rhsT)
    return nullptr;

  if (lhsT.isa<IntT>()) {
    return intOp(lhs, rhs, [](ConstantIntAttr lhsInt, ConstantIntAttr rhsInt) {
      return ConstantIntAttr::get(lhsInt.getContext(),
                                  lhsInt.getValue() & rhsInt.getValue(),
                                  lhsInt.getBitWidth(), lhsInt.isSigned());
    });
  }
  return nullptr;
}

static ConstantAttr bitOr(ConstantAttr lhs, ConstantAttr rhs) {
  auto lhsT = lhs.getType();
  auto rhsT = rhs.getType();
  if (lhsT != rhsT)
    return nullptr;

  if (lhsT.isa<IntT>()) {
    return intOp(lhs, rhs, [](ConstantIntAttr lhsInt, ConstantIntAttr rhsInt) {
      return ConstantIntAttr::get(lhsInt.getContext(),
                                  lhsInt.getValue() | rhsInt.getValue(),
                                  lhsInt.getBitWidth(), lhsInt.isSigned());
    });
  }
  return nullptr;
}

static ConstantAttr bitXor(ConstantAttr lhs, ConstantAttr rhs) {
  auto lhsT = lhs.getType();
  auto rhsT = rhs.getType();
  if (lhsT != rhsT)
    return nullptr;

  if (lhsT.isa<IntT>()) {
    return intOp(lhs, rhs, [](ConstantIntAttr lhsInt, ConstantIntAttr rhsInt) {
      return ConstantIntAttr::get(lhsInt.getContext(),
                                  lhsInt.getValue() ^ rhsInt.getValue(),
                                  lhsInt.getBitWidth(), lhsInt.isSigned());
    });
  }
  return nullptr;
}

static ConstantAttr shl(ConstantAttr lhs, ConstantAttr rhs) {
  auto lhsT = lhs.getType();
  auto rhsT = rhs.getType();
  if (lhsT != rhsT)
    return nullptr;

  if (lhsT.isa<IntT>()) {
    return intOp(
        lhs, rhs,
        [](ConstantIntAttr lhsInt, ConstantIntAttr rhsInt) -> ConstantIntAttr {
          if (rhsInt.getType().cast<IntT>().isSigned()) {
            if (static_cast<std::int64_t>(rhsInt.getValue()) < 0)
              return nullptr;
          }
          return ConstantIntAttr::get(lhsInt.getContext(),
                                      lhsInt.getValue() << rhsInt.getValue(),
                                      lhsInt.getBitWidth(), lhsInt.isSigned());
        });
  }
  return nullptr;
}

static ConstantAttr shr(ConstantAttr lhs, ConstantAttr rhs) {
  auto lhsT = lhs.getType();
  auto rhsT = rhs.getType();
  if (lhsT != rhsT)
    return nullptr;

  if (lhsT.isa<IntT>()) {
    return intOp(
        lhs, rhs,
        [](ConstantIntAttr lhsInt, ConstantIntAttr rhsInt) -> ConstantIntAttr {
          if (rhsInt.getType().cast<IntT>().isSigned()) {
            if (static_cast<std::int64_t>(rhsInt.getValue()) < 0)
              return nullptr; // Negative shift is not defined
          }
          return ConstantIntAttr::get(lhsInt.getContext(),
                                      lhsInt.getValue() >> rhsInt.getValue(),
                                      lhsInt.getBitWidth(), lhsInt.isSigned());
        });
  }
  return nullptr;
}

static ConstantAttr eq(ConstantAttr lhs, ConstantAttr rhs) {
  auto lhsT = lhs.getType();
  auto rhsT = rhs.getType();
  if (lhsT != rhsT)
    return nullptr;

  if (lhsT.isa<IntT>()) {
    return intOp(lhs, rhs, [](ConstantIntAttr lhsInt, ConstantIntAttr rhsInt) {
      return ConstantIntAttr::get(
          lhsInt.getContext(), lhsInt.getValue() == rhsInt.getValue() ? 1 : 0,
          1, true);
    });
  } else if (lhsT.isa<FloatT>()) {
    return floatOp(
        lhs, rhs, [](ConstantFloatAttr lhsFloat, ConstantFloatAttr rhsFloat) {
          return ConstantIntAttr::get(
              lhsFloat.getContext(),
              lhsFloat.getValue() == rhsFloat.getValue() ? 1 : 0, 1, true);
        });
  }
  return nullptr;
}

static ConstantAttr ne(ConstantAttr lhs, ConstantAttr rhs) {
  auto lhsT = lhs.getType();
  auto rhsT = rhs.getType();
  if (lhsT != rhsT)
    return nullptr;

  if (lhsT.isa<IntT>()) {
    return intOp(lhs, rhs, [](ConstantIntAttr lhsInt, ConstantIntAttr rhsInt) {
      return ConstantIntAttr::get(
          lhsInt.getContext(), lhsInt.getValue() != rhsInt.getValue() ? 1 : 0,
          1, true);
    });
  } else if (lhsT.isa<FloatT>()) {
    return floatOp(
        lhs, rhs, [](ConstantFloatAttr lhsFloat, ConstantFloatAttr rhsFloat) {
          return ConstantIntAttr::get(
              lhsFloat.getContext(),
              lhsFloat.getValue() != rhsFloat.getValue() ? 1 : 0, 1, true);
        });
  }
  return nullptr;
}

static ConstantAttr lt(ConstantAttr lhs, ConstantAttr rhs) {
  auto lhsT = lhs.getType();
  auto rhsT = rhs.getType();
  if (lhsT != rhsT)
    return nullptr;

  if (lhsT.isa<IntT>()) {
    return intOp(lhs, rhs, [](ConstantIntAttr lhsInt, ConstantIntAttr rhsInt) {
      if (lhsInt.isSigned())
        return ConstantIntAttr::get(
            lhsInt.getContext(),
            static_cast<std::int64_t>(lhsInt.getValue()) <
                    static_cast<std::int64_t>(rhsInt.getValue())
                ? 1
                : 0,
            1, true);
      return ConstantIntAttr::get(lhsInt.getContext(),
                                  lhsInt.getValue() < rhsInt.getValue() ? 1 : 0,
                                  1, true);
    });
  } else if (lhsT.isa<FloatT>()) {
    return floatOp(
        lhs, rhs, [](ConstantFloatAttr lhsFloat, ConstantFloatAttr rhsFloat) {
          return ConstantIntAttr::get(
              lhsFloat.getContext(),
              lhsFloat.getValue() < rhsFloat.getValue() ? 1 : 0, 1, true);
        });
  }
  return nullptr;
}

static ConstantAttr le(ConstantAttr lhs, ConstantAttr rhs) {
  auto lhsT = lhs.getType();
  auto rhsT = rhs.getType();
  if (lhsT != rhsT)
    return nullptr;

  if (lhsT.isa<IntT>()) {
    return intOp(lhs, rhs, [](ConstantIntAttr lhsInt, ConstantIntAttr rhsInt) {
      return ConstantIntAttr::get(
          lhsInt.getContext(), lhsInt.getValue() <= rhsInt.getValue() ? 1 : 0,
          1, true);
    });
  } else if (lhsT.isa<FloatT>()) {
    return floatOp(
        lhs, rhs, [](ConstantFloatAttr lhsFloat, ConstantFloatAttr rhsFloat) {
          return ConstantIntAttr::get(
              lhsFloat.getContext(),
              lhsFloat.getValue() <= rhsFloat.getValue() ? 1 : 0, 1, true);
        });
  }
  return nullptr;
}

static ConstantAttr minus(ConstantAttr value) {
  auto valueT = value.getType();
  if (valueT.isa<IntT>()) {
    auto intValue = value.cast<ConstantIntAttr>();
    return ConstantIntAttr::get(intValue.getContext(), -intValue.getValue(),
                                intValue.getBitWidth(), intValue.isSigned());
  } else if (valueT.isa<FloatT>()) {
    auto floatValue = value.dyn_cast<ConstantFloatAttr>();
    if (!floatValue) {
      auto strFloat = value.cast<ConstantStringFloatAttr>();
      floatValue = strFloat.convertToFloatAttr();
    }
    return ConstantFloatAttr::get(floatValue.getContext(),
                                  -floatValue.getValue());
  }

  return nullptr;
}

// ~
static ConstantAttr negate(ConstantAttr value) {
  auto valueT = value.getType();
  if (valueT.isa<IntT>()) {
    auto intValue = value.cast<ConstantIntAttr>();
    return ConstantIntAttr::get(intValue.getContext(), ~intValue.getValue(),
                                intValue.getBitWidth(), intValue.isSigned());
  }
  return nullptr;
}

} // namespace fold

class BinaryFold : public InstConversionPattern<inst::Binary> {
public:
  BinaryFold() : InstConversionPattern() {}

  utils::LogicalResult matchAndRewrite(IRRewriter &rewriter,
                                       inst::Binary::Adaptor adaptor,
                                       inst::Binary binary) override {
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();

    auto lhsConstant = lhs.getDefiningInst<inst::Constant>();
    if (!lhsConstant)
      return utils::LogicalResult::failure();
    auto rhsConstant = rhs.getDefiningInst<inst::Constant>();
    if (!rhsConstant)
      return utils::LogicalResult::failure();

    auto lhsAttr = lhsConstant.getValue();
    auto rhsAttr = rhsConstant.getValue();

    ConstantAttr result;
    switch (binary.getOpKind()) {
    case inst::Binary::Add:
      result = fold::add(lhsAttr, rhsAttr);
      break;
    case inst::Binary::Sub:
      result = fold::sub(lhsAttr, rhsAttr);
      break;
    case inst::Binary::Mul:
      result = fold::mul(lhsAttr, rhsAttr);
      break;
    case inst::Binary::Div:
      result = fold::div(lhsAttr, rhsAttr);
      break;
    case inst::Binary::Mod:
      result = fold::mod(lhsAttr, rhsAttr);
      break;
    case inst::Binary::BitAnd:
      result = fold::bitAnd(lhsAttr, rhsAttr);
      break;
    case inst::Binary::BitOr:
      result = fold::bitOr(lhsAttr, rhsAttr);
      break;
    case inst::Binary::BitXor:
      result = fold::bitXor(lhsAttr, rhsAttr);
      break;
    case inst::Binary::Shl:
      result = fold::shl(lhsAttr, rhsAttr);
      break;
    case inst::Binary::Shr:
      result = fold::shr(lhsAttr, rhsAttr);
      break;
    case inst::Binary::Eq:
      result = fold::eq(lhsAttr, rhsAttr);
      break;
    case inst::Binary::Ne:
      result = fold::ne(lhsAttr, rhsAttr);
      break;
    case inst::Binary::Lt:
      result = fold::lt(lhsAttr, rhsAttr);
      break;
    case inst::Binary::Le:
      result = fold::le(lhsAttr, rhsAttr);
      break;
    case inst::Binary::Gt:
      result = fold::lt(
          rhsAttr, lhsAttr); // Gt is equivalent to Lt with swapped operands
      break;
    case inst::Binary::Ge:
      result = fold::le(
          rhsAttr, lhsAttr); // Ge is equivalent to Le with swapped operands
      break;
    }

    if (!result)
      return utils::LogicalResult::failure();

    IRBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(
        binary.getParentBlock()->getParentIR()->getConstantBlock());
    Value replaceValue =
        rewriter.create<inst::Constant>(binary.getRange(), result);

    rewriter.replaceInst(binary.getStorage(), replaceValue);
    return utils::LogicalResult::success(); // match success
  }
};

class UnaryFold : public InstConversionPattern<inst::Unary> {
public:
  UnaryFold() : InstConversionPattern() {}

  utils::LogicalResult matchAndRewrite(IRRewriter &rewriter,
                                       inst::Unary::Adaptor adaptor,
                                       inst::Unary unary) override {
    auto operand = adaptor.getValue();

    auto operandConstant = operand.getDefiningInst<inst::Constant>();
    if (!operandConstant)
      return utils::LogicalResult::failure();

    auto operandAttr = operandConstant.getValue();
    ConstantAttr result;
    switch (unary.getOpKind()) {
    case inst::Unary::Minus:
      result = fold::minus(operandAttr);
      break;
    case inst::Unary::Negate:
      result = fold::negate(operandAttr);
      break;
    case inst::Unary::Plus:
      result = operandAttr; // Unary plus does not change the value
      break;
    }

    if (!result)
      return utils::LogicalResult::failure();

    IRBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(
        unary.getParentBlock()->getParentIR()->getConstantBlock());
    Value replaceValue =
        rewriter.create<inst::Constant>(unary.getRange(), result);
    rewriter.replaceInst(unary.getStorage(), replaceValue);
    return utils::LogicalResult::success(); // match success
  }
};

PassResult InstructionFold::run(Module *module) {
  PatternSet patterns;
  patterns.addPatterns<BinaryFold, UnaryFold>();
  applyPatternConversion(module, patterns);
  return PassResult::success(); // pass success
}

} // namespace kecc::ir
