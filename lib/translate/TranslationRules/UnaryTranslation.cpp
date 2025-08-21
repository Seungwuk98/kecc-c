#include "kecc/asm/AsmInstruction.h"
#include "kecc/asm/Register.h"
#include "kecc/ir/IRAttributes.h"
#include "kecc/ir/IRInstructions.h"
#include "kecc/translate/IRTranslater.h"
#include "kecc/utils/LogicalResult.h"

namespace kecc {

namespace {

static utils::LogicalResult translatePlus(as::AsmBuilder &builder,
                                          FunctionTranslater &translater,
                                          ir::inst::Unary inst);

static utils::LogicalResult translateMinus(as::AsmBuilder &builder,
                                           FunctionTranslater &translater,
                                           ir::inst::Unary inst);

static utils::LogicalResult translateNegate(as::AsmBuilder &builder,
                                            FunctionTranslater &translater,
                                            ir::inst::Unary inst);

static utils::LogicalResult (*unaryTranslationFunctions[])(as::AsmBuilder &,
                                                           FunctionTranslater &,
                                                           ir::inst::Unary) = {
    translatePlus,   // Unary::Plus
    translateMinus,  // Unary::Minus
    translateNegate, // Unary::Negate
};

} // namespace

class UnaryTranslationRule
    : public InstructionTranslationRule<ir::inst::Unary> {
public:
  UnaryTranslationRule() {}

  utils::LogicalResult translate(as::AsmBuilder &builder,
                                 FunctionTranslater &translater,
                                 ir::inst::Unary inst) override;
};

utils::LogicalResult
UnaryTranslationRule::translate(as::AsmBuilder &builder,
                                FunctionTranslater &translater,
                                ir::inst::Unary inst) {
  auto opKind = inst.getOpKind();

  auto translateFunc = unaryTranslationFunctions[static_cast<int>(opKind)];
  return translateFunc(builder, translater, inst);
}

namespace {

utils::LogicalResult translatePlus(as::AsmBuilder &builder,
                                   FunctionTranslater &translater,
                                   ir::inst::Unary inst) {
  return utils::LogicalResult::success(); // nop
}

utils::LogicalResult translateMinus(as::AsmBuilder &builder,
                                    FunctionTranslater &translater,
                                    ir::inst::Unary inst) {

  const auto *operand = &inst.getValueAsOperand();

  as::Register rd = translater.getRegister(inst.getResult());

  if (operand->isConstant()) {
    assert(!operand->getType().isa<ir::FloatT>() &&
           "Minus operation on float constant is not supported.");
    auto constantAttr =
        operand->getDefiningInst<ir::inst::Constant>().getValue();

    auto minus = constantAttr.insertMinus();
    assert(minus && "Minus operation should be valid for constant");

    // addi zero imm
    auto *imm = getImmediate(minus);
    assert(imm && "Immediate should not be null");

    // imm must be 12-bit
    builder.create<as::itype::Addi>(rd, as::Register::zero(), imm,
                                    as::DataSize::half());
  } else {
    as::Register rs = translater.getOperandRegister(operand);

    auto dataSize = getDataSize(operand->getType());
    // sub rd, zero, rs
    builder.create<as::rtype::Sub>(rd, as::Register::zero(), rs, dataSize);
  }

  return utils::LogicalResult::success();
}

utils::LogicalResult translateNegate(as::AsmBuilder &builder,
                                     FunctionTranslater &translater,
                                     ir::inst::Unary inst) {
  const auto *operand = &inst.getValueAsOperand();
  as::Register rd = translater.getRegister(inst.getResult());

  assert(!operand->getType().isa<ir::FloatT>() &&
         "Negate operation is not supported for float types.");

  if (operand->isConstant()) {
    auto constantAttr =
        operand->getDefiningInst<ir::inst::Constant>().getValue();
    auto value = constantAttr.cast<ir::ConstantIntAttr>().getValue();
    value = ~value;

    auto *imm = getImmediate(static_cast<int64_t>(value));
    assert(imm && "Immediate should not be null");
    // addi rd, zero, imm
    builder.create<as::itype::Addi>(rd, as::Register::zero(), imm,
                                    as::DataSize::half());
  } else {
    as::Register rs = translater.getOperandRegister(operand);
    // not rd, rs

    builder.create<as::pseudo::Not>(rd, rs);
  }
  return utils::LogicalResult::success();
}

} // namespace
} // namespace kecc
