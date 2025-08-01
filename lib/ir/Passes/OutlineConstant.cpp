#include "kecc/ir/IRAttributes.h"
#include "kecc/ir/IRBuilder.h"
#include "kecc/ir/IRInstructions.h"
#include "kecc/ir/IRTransforms.h"
#include "kecc/ir/Instruction.h"
#include "kecc/ir/PatternMatch.h"
#include "llvm/ADT/TypeSwitch.h"

namespace kecc::ir {

namespace {

#define U_UPPER_IMM12 ((1 << 12) - 1)
#define I_UPPER_IMM12 ((1 << 11) - 1)
#define I_LOWER_IMM12 (-(1 << 11))

struct InlineConstantState {
  enum Kind {
    NonConstant,
    CanbeInline,
    Outline,
  };

  static InlineConstantState nonConstant() {
    return InlineConstantState(NonConstant);
  }

  static InlineConstantState canbeInline() {
    return InlineConstantState(CanbeInline);
  }

  static InlineConstantState outline() { return InlineConstantState(Outline); }

  bool isNonConstant() const { return kind == NonConstant; }
  bool isCanbeInline() const { return kind == CanbeInline; }
  bool isOutline() const { return kind == Outline; }

  Kind kind;
};

static InlineConstantState getInlineConstantState(Value value) {
  if (auto constant = value.getDefiningInst<inst::Constant>()) {
    auto constAttr = constant.getValue();
    auto canbeInline = llvm::TypeSwitch<ConstantAttr, bool>(constAttr)
                           .Case([&](ConstantIntAttr intAttr) -> bool {
                             auto value = intAttr.getValue();
                             auto bitWidth = intAttr.getIntType().getBitWidth();
                             value = value & ((1ULL << bitWidth) - 1);

                             if (intAttr.getIntType().isSigned()) {
                               std::int64_t signedValue = value;
                               return signedValue >= I_LOWER_IMM12 &&
                                      signedValue <= I_UPPER_IMM12;
                             } else {
                               return value <= U_UPPER_IMM12;
                             }
                           })
                           .Default([&](ConstantAttr) { return false; });
    return canbeInline ? InlineConstantState::canbeInline()
                       : InlineConstantState::outline();
  }
  return InlineConstantState::nonConstant();
}

static void createOutline(IRRewriter &rewriter, InstructionStorage *inst,
                          Value constant,
                          llvm::function_ref<void(Value)> operandSetter) {
  IRBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointBeforeInst(inst);
  auto outlineConstant =
      rewriter.create<inst::OutlineConstant>(inst->getRange(), constant);
  operandSetter(outlineConstant);
}

class ConvertBinary : public InstPattern<inst::Binary> {
public:
  ConvertBinary() : InstPattern() {}

  utils::LogicalResult matchAndRewrite(IRRewriter &rewriter,
                                       inst::Binary binary) override {

    auto lhs = binary.getLhs();
    auto rhs = binary.getRhs();
    auto lhsCS = getInlineConstantState(lhs);
    auto rhsCS = getInlineConstantState(rhs);

    auto lhsSetter = [binary](Value value) {
      binary.getStorage()->setOperand(0, value);
    };
    auto rhsSetter = [binary](Value value) {
      binary.getStorage()->setOperand(1, value);
    };

    bool matched = false;
    bool notified = false;

    auto createOutlineAndMatch = [&](Value value,
                                     llvm::function_ref<void(Value)> setter) {
      matched = true;
      if (!notified) {
        rewriter.notifyStartUpdate(binary.getStorage());
        notified = true;
      }
      createOutline(rewriter, binary.getStorage(), value, setter);
    };
    switch (binary.getOpKind()) {
    case inst::Binary::Add:
    case inst::Binary::Sub:
    case inst::Binary::BitAnd:
    case inst::Binary::BitOr:
    case inst::Binary::BitXor:
      if (lhsCS.isCanbeInline()) {
        if (!rhsCS.isNonConstant())
          createOutlineAndMatch(rhs, rhsSetter);
      } else if (rhsCS.isCanbeInline()) {
        if (!lhsCS.isNonConstant())
          createOutlineAndMatch(lhs, lhsSetter);
      } else {
        if (!lhsCS.isNonConstant())
          createOutlineAndMatch(lhs, lhsSetter);
        if (!rhsCS.isNonConstant())
          createOutlineAndMatch(rhs, lhsSetter);
      }
    case inst::Binary::Shl:
    case inst::Binary::Shr:
      if (rhsCS.isOutline()) {
        createOutlineAndMatch(rhs, rhsSetter);
      }
    case inst::Binary::Mul:
    case inst::Binary::Div:
    case inst::Binary::Mod:
    case inst::Binary::Eq:
    case inst::Binary::Ne:
    case inst::Binary::Lt:
    case inst::Binary::Le:
    case inst::Binary::Gt:
    case inst::Binary::Ge:
      if (lhsCS.isOutline())
        createOutlineAndMatch(lhs, lhsSetter);
      else if (rhsCS.isOutline())
        createOutlineAndMatch(rhs, rhsSetter);
    }

    return matched ? utils::LogicalResult::success()
                   : utils::LogicalResult::failure(); // match fail
  }
};

class ConvertInstructions : public Pattern {
public:
  ConvertInstructions() : Pattern(1, PatternId::getGeneral()) {}

  utils::LogicalResult matchAndRewrite(IRRewriter &rewriter,
                                       InstructionStorage *inst) override {
    if (inst->getDefiningInst<inst::Binary>() ||
        inst->getDefiningInst<inst::OutlineConstant>())
      return utils::LogicalResult::failure();

    bool matched = false;
    bool notified = false;
    for (auto [idx, value] : llvm::enumerate(inst->getOperands())) {
      auto inlineState = getInlineConstantState(value);
      if (!inlineState.isNonConstant()) {
        if (!notified) {
          rewriter.notifyStartUpdate(inst);
          notified = true;
        }
        createOutline(rewriter, inst, value,
                      [&](Value newValue) { inst->setOperand(idx, newValue); });
        matched = true;
      }
    }

    for (auto [idx, value] : llvm::enumerate(inst->getJumpArgs())) {
      auto jumpArgState = value->getAsState();
      bool argMatched = false;
      for (auto [argIdx, arg] : llvm::enumerate(jumpArgState.getArgs())) {
        auto inlineState = getInlineConstantState(arg);
        if (!inlineState.isNonConstant()) {
          createOutline(rewriter, inst, arg, [&](Value newValue) {
            jumpArgState.setArg(argIdx, newValue);
          });
          argMatched = true;
        }
      }
      if (argMatched) {
        if (!notified) {
          rewriter.notifyStartUpdate(inst);
          notified = true;
        }
        inst->setJumpArg(idx, jumpArgState);
        matched = true;
      }
    }

    return matched ? utils::LogicalResult::success()
                   : utils::LogicalResult::failure(); // match fail
  }
};

} // namespace

void OutlineConstantPass::init(Module *module) {
  if (!module->getContext()->isRegisteredInst<inst::OutlineConstant>()) {
    module->getContext()->registerInst<inst::OutlineConstant>();
  }
}

PassResult OutlineConstantPass::run(Module *module) {
  PatternSet patterns;
  patterns.addPatterns<ConvertBinary, ConvertInstructions>();

  applyPatternConversion(module, patterns);
  return PassResult::success();
}

} // namespace kecc::ir
