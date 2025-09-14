#include "kecc/ir/IRAttributes.h"
#include "kecc/ir/IRBuilder.h"
#include "kecc/ir/IRInstructions.h"
#include "kecc/ir/IRTransforms.h"
#include "kecc/ir/Instruction.h"
#include "kecc/ir/PatternMatch.h"
#include "kecc/translate/TranslateConstants.h"
#include "llvm/ADT/TypeSwitch.h"

namespace kecc::ir {

namespace {

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
    auto canbeInline =
        llvm::TypeSwitch<ConstantAttr, bool>(constAttr)
            .Case([&](ConstantIntAttr intAttr) -> bool {
              auto value = intAttr.getValue();
              std::int64_t signedValue = value;
              return signedValue >= MIN_INT_12 && signedValue <= MAX_INT_12;
            })
            .Default([&](ConstantAttr) { return false; });
    return canbeInline ? InlineConstantState::canbeInline()
                       : InlineConstantState::outline();
  }
  return InlineConstantState::nonConstant();
}

static Value createOutline(IRRewriter &rewriter, InstructionStorage *inst,
                           Value constant,
                           llvm::function_ref<void(Value)> operandSetter) {
  IRBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointBeforeInst(inst);
  auto outlineConstant =
      rewriter.create<inst::OutlineConstant>(inst->getRange(), constant);
  operandSetter(outlineConstant);
  return outlineConstant;
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
      break;
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
      if (!lhsCS.isNonConstant())
        createOutlineAndMatch(lhs, lhsSetter);
      if (!rhsCS.isNonConstant())
        createOutlineAndMatch(rhs, rhsSetter);
      break;
    }

    return matched ? utils::LogicalResult::success()
                   : utils::LogicalResult::failure(); // match fail
  }
};

class ConvertGep : public InstPattern<inst::Gep> {
public:
  ConvertGep() : InstPattern() {}

  utils::LogicalResult matchAndRewrite(IRRewriter &rewriter,
                                       inst::Gep gep) override {
    bool matched = false;
    auto baseCS = getInlineConstantState(gep.getBasePointer());
    if (!baseCS.isNonConstant()) {
      rewriter.notifyStartUpdate(gep.getStorage());
      createOutline(
          rewriter, gep.getStorage(), gep.getBasePointer(),
          [&](Value newValue) { gep.getStorage()->setOperand(0, newValue); });
      matched = true;
    }

    auto offsetCS = getInlineConstantState(gep.getOffset());
    if (offsetCS.isOutline()) {
      if (!matched)
        rewriter.notifyStartUpdate(gep.getStorage());
      createOutline(
          rewriter, gep.getStorage(), gep.getOffset(),
          [&](Value newValue) { gep.getStorage()->setOperand(1, newValue); });
      matched = true;
    }

    return matched ? utils::LogicalResult::success()
                   : utils::LogicalResult::failure(); // match fail
  }
};

static bool
outlineJumpArgs(IRRewriter &rewriter, InstructionStorage *inst, bool notified,
                llvm::DenseMap<ConstantAttr, Value> &constantCache) {
  bool matched = false;
  for (auto [idx, value] : llvm::enumerate(inst->getJumpArgs())) {
    auto jumpArgState = value->getAsState();
    bool argMatched = false;
    for (auto [argIdx, arg] : llvm::enumerate(jumpArgState.getArgs())) {
      auto inlineState = getInlineConstantState(arg);
      if (!inlineState.isNonConstant()) {

        auto constantAttr = arg.getDefiningInst<inst::Constant>().getValue();
        if (constantAttr.isa<ConstantUnitAttr>())
          continue; // Unit constant should not be outlined
        if (!notified) {
          rewriter.notifyStartUpdate(inst);
          notified = true;
        }

        if (auto it = constantCache.find(constantAttr);
            it != constantCache.end()) {
          jumpArgState.setArg(argIdx, it->second);
        } else {
          auto outlineConstant =
              createOutline(rewriter, inst, arg, [&](Value newValue) {
                jumpArgState.setArg(argIdx, newValue);
              });
          if (!constantAttr.isa<ConstantUndefAttr>())
            constantCache[constantAttr] = outlineConstant;
        }

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
  return matched;
}

class ConvertSwitch : public InstPattern<inst::Switch> {
public:
  ConvertSwitch() : InstPattern() {}

  utils::LogicalResult matchAndRewrite(IRRewriter &rewriter,
                                       inst::Switch switchInst) override {

    auto valueCS = getInlineConstantState(switchInst.getValue());
    bool matched = false;
    llvm::DenseMap<ConstantAttr, Value> constantCache;
    if (!valueCS.isNonConstant()) {
      rewriter.notifyStartUpdate(switchInst.getStorage());
      auto constant =
          switchInst.getValue().getDefiningInst<inst::Constant>().getValue();
      auto outline =
          createOutline(rewriter, switchInst.getStorage(),
                        switchInst.getValue(), [&](Value newValue) {
                          switchInst.getStorage()->setOperand(0, newValue);
                        });
      constantCache.try_emplace(constant, outline);
      matched = true;
    }

    matched |= outlineJumpArgs(rewriter, switchInst.getStorage(), matched,
                               constantCache);

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
        inst->getDefiningInst<inst::Gep>() ||
        inst->getDefiningInst<inst::OutlineConstant>() ||
        inst->getDefiningInst<inst::Switch>())
      return utils::LogicalResult::failure();

    bool matched = false;
    bool notified = false;

    llvm::DenseMap<ConstantAttr, Value> constantCache;
    for (auto [idx, value] : llvm::enumerate(inst->getOperands())) {
      auto inlineState = getInlineConstantState(value);
      if (!inlineState.isNonConstant()) {
        auto constantAttr = value.getDefiningInst<inst::Constant>().getValue();
        if (constantAttr.isa<ConstantUnitAttr>())
          continue; // Unit constant should not be outlined
        if (!notified) {
          rewriter.notifyStartUpdate(inst);
          notified = true;
        }
        if (auto it = constantCache.find(constantAttr);
            it != constantCache.end()) {
          inst->setOperand(idx, it->second);
        } else {
          auto outlineConstant =
              createOutline(rewriter, inst, value, [&](Value newValue) {
                inst->setOperand(idx, newValue);
              });
          if (!constantAttr.isa<ConstantUndefAttr>())
            // Undef should not be cached
            // because it cause unnecessary copies when we move registers
            constantCache[constantAttr] = outlineConstant;
        }

        matched = true;
      }
    }

    matched |= outlineJumpArgs(rewriter, inst, notified, constantCache);
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
  patterns.addPatterns<ConvertBinary, ConvertGep, ConvertSwitch,
                       ConvertInstructions>();

  applyPatternConversion(module, patterns);

  return PassResult::success();
}

} // namespace kecc::ir
