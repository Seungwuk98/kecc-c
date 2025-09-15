#include "kecc/asm/Asm.h"
#include "kecc/asm/AsmInstruction.h"
#include "kecc/asm/Register.h"
#include "kecc/ir/IRInstructions.h"
#include "kecc/ir/IRTypes.h"
#include "kecc/translate/IRTranslater.h"
#include "kecc/translate/IntermediateInsts.h"
#include "kecc/translate/TranslateContext.h"
#include "kecc/utils/LogicalResult.h"

namespace kecc {

//===----------------------------------------------------------------------===//
/// Binary Instruction Translation
//===----------------------------------------------------------------------===//
namespace {
static utils::LogicalResult translateAdd(as::AsmBuilder &builder,
                                         FunctionTranslater &translater,
                                         ir::inst::Binary inst);
static utils::LogicalResult translateSub(as::AsmBuilder &builder,
                                         FunctionTranslater &translater,
                                         ir::inst::Binary inst);
static utils::LogicalResult translateMul(as::AsmBuilder &builder,
                                         FunctionTranslater &translater,
                                         ir::inst::Binary inst);
static utils::LogicalResult translateDiv(as::AsmBuilder &builder,
                                         FunctionTranslater &translater,
                                         ir::inst::Binary inst);
static utils::LogicalResult translateMod(as::AsmBuilder &builder,
                                         FunctionTranslater &translater,
                                         ir::inst::Binary inst);
static utils::LogicalResult translateAnd(as::AsmBuilder &builder,
                                         FunctionTranslater &translater,
                                         ir::inst::Binary inst);
static utils::LogicalResult translateOr(as::AsmBuilder &builder,
                                        FunctionTranslater &translater,
                                        ir::inst::Binary inst);
static utils::LogicalResult translateXor(as::AsmBuilder &builder,
                                         FunctionTranslater &translater,
                                         ir::inst::Binary inst);
static utils::LogicalResult translateShl(as::AsmBuilder &builder,
                                         FunctionTranslater &translater,
                                         ir::inst::Binary inst);
static utils::LogicalResult translateShr(as::AsmBuilder &builder,
                                         FunctionTranslater &translater,
                                         ir::inst::Binary inst);
static utils::LogicalResult translateEq(as::AsmBuilder &builder,
                                        FunctionTranslater &translater,
                                        ir::inst::Binary inst);
static utils::LogicalResult translateNeq(as::AsmBuilder &builder,
                                         FunctionTranslater &translater,
                                         ir::inst::Binary inst);
static utils::LogicalResult translateLt(as::AsmBuilder &builder,
                                        FunctionTranslater &translater,
                                        ir::inst::Binary inst);
static utils::LogicalResult translateLe(as::AsmBuilder &builder,
                                        FunctionTranslater &translater,
                                        ir::inst::Binary inst);
static utils::LogicalResult translateGt(as::AsmBuilder &builder,
                                        FunctionTranslater &translater,
                                        ir::inst::Binary inst);
static utils::LogicalResult translateGe(as::AsmBuilder &builder,
                                        FunctionTranslater &translater,
                                        ir::inst::Binary inst);

static utils::LogicalResult (*binaryTranslationFunctions[])(
    as::AsmBuilder &, FunctionTranslater &,
    ir::inst::Binary) = {translateAdd, translateSub, translateMul, translateDiv,
                         translateMod, translateAnd, translateOr,  translateXor,
                         translateShl, translateShr, translateEq,  translateNeq,
                         translateLt,  translateLe,  translateGt,  translateGe};

} // namespace

class BinaryTranslationRule
    : public InstructionTranslationRule<ir::inst::Binary> {
public:
  BinaryTranslationRule() {}

  utils::LogicalResult translate(as::AsmBuilder &builder,
                                 FunctionTranslater &translater,
                                 ir::inst::Binary inst) override;
};

utils::LogicalResult
BinaryTranslationRule::translate(as::AsmBuilder &builder,
                                 FunctionTranslater &translater,
                                 ir::inst::Binary inst) {
  auto opKind = inst.getOpKind();

  auto translateFunc = binaryTranslationFunctions[static_cast<int>(opKind)];
  return translateFunc(builder, translater, inst);
}

namespace {

as::Instruction *createLogicalNot(as::AsmBuilder &builder, as::Register rd) {
  // xori rd, rd, 1
  auto *imm = new as::ValueImmediate(1);
  return builder.create<as::itype::Xori>(rd, rd, imm);
}

void shiftDataByType(as::AsmBuilder &builder, ir::IntT intT, as::Register rd) {
  if (intT.getBitWidth() == 1) {
    builder.create<as::itype::Andi>(rd, rd, getImmediate(1));
    return;
  }

  int shiftSize =
      (int)intT.getContext()->getArchitectureBitSize() - intT.getBitWidth();
  if (shiftSize <= 32)
    return;

  if (intT.isSigned()) {
    builder.create<as::itype::Slli>(rd, rd, getImmediate(shiftSize),
                                    as::DataSize::doubleWord());
    builder.create<as::itype::Srai>(rd, rd, getImmediate(shiftSize),
                                    as::DataSize::doubleWord());
  } else {
    if (intT.getBitWidth() <= 12) {
      builder.create<as::itype::Andi>(
          rd, rd, getImmediate((1LL << intT.getBitWidth()) - 1));
    } else {
      builder.create<as::itype::Slli>(rd, rd, getImmediate(shiftSize),
                                      as::DataSize::doubleWord());
      builder.create<as::itype::Srli>(rd, rd, getImmediate(shiftSize),
                                      as::DataSize::doubleWord());
    }
  }
}

utils::LogicalResult translateIntAdd(as::AsmBuilder &builder,
                                     FunctionTranslater &translater,
                                     ir::inst::Binary inst) {
  const auto *lhs = &inst.getLhsAsOperand();
  const auto *rhs = &inst.getRhsAsOperand();
  auto dest = inst.getResult();
  auto rd = translater.getRegister(dest);
  auto dataSize = getDataSize(lhs->getType());

  assert((!lhs->isConstant() || !rhs->isConstant()) &&
         "Binary addition must have at least one non-constant operand. "
         "It is guaranteed by `OutlineConstant` pass.");

  if (lhs->isConstant() | rhs->isConstant()) {
    // use itype
    if (lhs->isConstant())
      std::swap(lhs, rhs);

    auto *rhsImm = getImmediate(rhs->getDefiningInst<ir::inst::Constant>());
    assert(rhsImm && "Expected a constant immediate for rhs operand");

    auto lhsReg = translater.getOperandRegister(lhs);
    if (rhsImm->isZero()) {
      // mv dest, lhs
      if (rd != lhsReg)
        builder.create<as::pseudo::Mv>(rd, lhsReg);
      return utils::LogicalResult::success();
    }

    builder.create<as::itype::Addi>(rd, lhsReg, rhsImm, dataSize);
  } else {
    // use rtype

    auto lhsReg = translater.getOperandRegister(lhs);
    auto rhsReg = translater.getOperandRegister(rhs);

    builder.create<as::rtype::Add>(rd, lhsReg, rhsReg, dataSize);
  }
  shiftDataByType(builder, lhs->getType().cast<ir::IntT>(), rd);
  return utils::LogicalResult::success();
}

utils::LogicalResult translateFloatAdd(as::AsmBuilder &builder,
                                       FunctionTranslater &translater,
                                       ir::inst::Binary inst) {
  const auto *lhs = &inst.getLhsAsOperand();
  const auto *rhs = &inst.getRhsAsOperand();
  auto dest = inst.getResult();
  auto rd = translater.getRegister(dest);
  auto dataSize = getDataSize(lhs->getType());

  assert(!lhs->isConstant() && !rhs->isConstant() &&
         "Binary float addition operands must not be constant. "
         "It is guaranteed by `OutlineConstant` pass.");

  auto lhsReg = translater.getOperandRegister(lhs);
  auto rhsReg = translater.getOperandRegister(rhs);
  builder.create<as::rtype::Fadd>(rd, lhsReg, rhsReg, dataSize);
  return utils::LogicalResult::success();
}

utils::LogicalResult translateAdd(as::AsmBuilder &builder,
                                  FunctionTranslater &translater,
                                  ir::inst::Binary inst) {
  if (inst.getResult().getType().isa<ir::FloatT>()) {
    return translateFloatAdd(builder, translater, inst);
  } else {
    return translateIntAdd(builder, translater, inst);
  }
}

utils::LogicalResult translateIntSub(as::AsmBuilder &builder,
                                     FunctionTranslater &translater,
                                     ir::inst::Binary inst) {
  const auto *lhs = &inst.getLhsAsOperand();
  const auto *rhs = &inst.getRhsAsOperand();
  auto dest = inst.getResult();
  auto rd = translater.getRegister(dest);
  auto intT = lhs->getType().cast<ir::IntT>();
  auto dataSize = getDataSize(intT);

  assert(!lhs->isConstant() && "Binary subtraction lhs must not be constant. "
                               "It is guaranteed by `OutlineConstant` pass.");
  auto lhsReg = translater.getOperandRegister(lhs);

  if (rhs->isConstant()) {
    // use itype
    auto *rhsImm = getImmediate(rhs->getDefiningInst<ir::inst::Constant>());
    assert(rhsImm && "Expected a constant immediate.");
    if (rhsImm->isZero()) {
      // mv dest, lhs
      builder.create<as::pseudo::Mv>(rd, lhsReg);
      return utils::LogicalResult::success();
    }
    rhsImm->applyMinus();

    builder.create<as::itype::Addi>(rd, lhsReg, rhsImm, dataSize);
  } else {
    // use rtype
    auto rhsReg = translater.getOperandRegister(rhs);
    builder.create<as::rtype::Sub>(rd, lhsReg, rhsReg, dataSize);
  }

  shiftDataByType(builder, lhs->getType().cast<ir::IntT>(), rd);

  return utils::LogicalResult::success();
}

utils::LogicalResult translateFloatSub(as::AsmBuilder &builder,
                                       FunctionTranslater &translater,
                                       ir::inst::Binary inst) {
  const auto *lhs = &inst.getLhsAsOperand();
  const auto *rhs = &inst.getRhsAsOperand();
  auto dest = inst.getResult();
  auto rd = translater.getRegister(dest);
  auto dataSize = getDataSize(lhs->getType());
  assert(!lhs->isConstant() && !rhs->isConstant() &&
         "Binary float subtraction operands must not be constant. "
         "It is guaranteed by `OutlineConstant` pass.");
  auto lhsReg = translater.getOperandRegister(lhs);
  auto rhsReg = translater.getOperandRegister(rhs);
  builder.create<as::rtype::Fsub>(rd, lhsReg, rhsReg, dataSize);
  return utils::LogicalResult::success();
}

utils::LogicalResult translateSub(as::AsmBuilder &builder,
                                  FunctionTranslater &translater,
                                  ir::inst::Binary inst) {
  if (inst.getResult().getType().isa<ir::FloatT>()) {
    return translateFloatSub(builder, translater, inst);
  } else {
    return translateIntSub(builder, translater, inst);
  }
}

utils::LogicalResult translateMul(as::AsmBuilder &builder,
                                  FunctionTranslater &translater,
                                  ir::inst::Binary inst) {
  const auto *lhs = &inst.getLhsAsOperand();
  const auto *rhs = &inst.getRhsAsOperand();
  auto dest = inst.getResult();
  auto rd = translater.getRegister(dest);
  auto dataSize = getDataSize(lhs->getType());
  assert(!lhs->isConstant() && !rhs->isConstant() &&
         "Binary multiplication operands must not be constant. "
         "It is guaranteed by `OutlineConstant` pass.");

  auto lhsReg = translater.getOperandRegister(lhs);
  auto rhsReg = translater.getOperandRegister(rhs);

  if (lhs->getType().isa<ir::FloatT>())
    builder.create<as::rtype::Fmul>(rd, lhsReg, rhsReg, dataSize);
  else {
    builder.create<as::rtype::Mul>(rd, lhsReg, rhsReg, dataSize);
    shiftDataByType(builder, lhs->getType().cast<ir::IntT>(), rd);
  }
  return utils::LogicalResult::success();
}

utils::LogicalResult translateDiv(as::AsmBuilder &builder,
                                  FunctionTranslater &translater,
                                  ir::inst::Binary inst) {
  const auto *lhs = &inst.getLhsAsOperand();
  const auto *rhs = &inst.getRhsAsOperand();
  auto dest = inst.getResult();
  auto rd = translater.getRegister(dest);
  auto dataSize = getDataSize(lhs->getType());
  assert(!lhs->isConstant() && !rhs->isConstant() &&
         "Binary division operands must not be constant. "
         "It is guaranteed by `OutlineConstant` pass.");

  auto lhsReg = translater.getOperandRegister(lhs);
  auto rhsReg = translater.getOperandRegister(rhs);

  if (lhs->getType().isa<ir::FloatT>())
    builder.create<as::rtype::Fdiv>(rd, lhsReg, rhsReg, dataSize);
  else {
    auto intT = rhs->getType().cast<ir::IntT>();
    builder.create<as::rtype::Div>(rd, lhsReg, rhsReg, dataSize,
                                   intT.isSigned());
    shiftDataByType(builder, lhs->getType().cast<ir::IntT>(), rd);
  }

  return utils::LogicalResult::success();
}

utils::LogicalResult translateMod(as::AsmBuilder &builder,
                                  FunctionTranslater &translater,
                                  ir::inst::Binary inst) {
  const auto *lhs = &inst.getLhsAsOperand();
  const auto *rhs = &inst.getRhsAsOperand();
  auto dest = inst.getResult();
  auto rd = translater.getRegister(dest);
  auto dataSize = getDataSize(lhs->getType());
  assert(!lhs->isConstant() && !rhs->isConstant() &&
         "Binary modulo operands must not be constant. "
         "It is guaranteed by `OutlineConstant` pass.");
  assert(!lhs->getType().isa<ir::FloatT>() &&
         "Binary modulo operands must not be float type. ");
  auto lhsReg = translater.getOperandRegister(lhs);
  auto rhsReg = translater.getOperandRegister(rhs);

  auto intT = rhs->getType().cast<ir::IntT>();

  builder.create<as::rtype::Rem>(rd, lhsReg, rhsReg, dataSize, intT.isSigned());
  return utils::LogicalResult::success();
}

utils::LogicalResult translateAnd(as::AsmBuilder &builder,
                                  FunctionTranslater &translater,
                                  ir::inst::Binary inst) {
  const auto *lhs = &inst.getLhsAsOperand();
  const auto *rhs = &inst.getRhsAsOperand();
  auto dest = inst.getResult();
  auto rd = translater.getRegister(dest);

  assert(!lhs->isConstant() ||
         !rhs->isConstant() &&
             "Binary AND must have at least one non-constant operand. "
             "It is guaranteed by `OutlineConstant` pass.");

  if (lhs->isConstant() | rhs->isConstant()) {
    // use itype
    if (lhs->isConstant())
      std::swap(lhs, rhs);

    auto *rhsImm = getImmediate(rhs->getDefiningInst<ir::inst::Constant>());
    assert(rhsImm && "Expected a constant immediate for rhs operand");
    auto lhsReg = translater.getOperandRegister(lhs);
    if (rhsImm->isZero()) {
      // mv dest, zero
      builder.create<as::pseudo::Mv>(rd, as::Register::zero());
      return utils::LogicalResult::success();
    }

    builder.create<as::itype::Andi>(rd, lhsReg, rhsImm);
    return utils::LogicalResult::success();
  } else {
    // use rtype
    auto lhsReg = translater.getOperandRegister(lhs);
    auto rhsReg = translater.getOperandRegister(rhs);

    builder.create<as::rtype::And>(rd, lhsReg, rhsReg);
    return utils::LogicalResult::success();
  }
}

utils::LogicalResult translateOr(as::AsmBuilder &builder,
                                 FunctionTranslater &translater,
                                 ir::inst::Binary inst) {
  const auto *lhs = &inst.getLhsAsOperand();
  const auto *rhs = &inst.getRhsAsOperand();
  auto dest = inst.getResult();
  auto rd = translater.getRegister(dest);
  assert(!lhs->isConstant() ||
         !rhs->isConstant() &&
             "Binary OR must have at least one non-constant operand. "
             "It is guaranteed by `OutlineConstant` pass.");

  if (lhs->isConstant() | rhs->isConstant()) {
    if (lhs->isConstant())
      std::swap(lhs, rhs);

    // use itype
    auto *rhsImm = getImmediate(rhs->getDefiningInst<ir::inst::Constant>());
    assert(rhsImm && "Expected a constant immediate for rhs operand");
    auto lhsReg = translater.getOperandRegister(lhs);
    if (rhsImm->isZero()) {
      // mv dest, lhs
      builder.create<as::pseudo::Mv>(rd, lhsReg);
      return utils::LogicalResult::success();
    }

    builder.create<as::itype::Ori>(rd, lhsReg, rhsImm);
    return utils::LogicalResult::success();
  } else {
    // use rtype
    auto lhsReg = translater.getOperandRegister(lhs);
    auto rhsReg = translater.getOperandRegister(rhs);

    builder.create<as::rtype::Or>(rd, lhsReg, rhsReg);
    return utils::LogicalResult::success();
  }
}

utils::LogicalResult translateXor(as::AsmBuilder &builder,
                                  FunctionTranslater &translater,
                                  ir::inst::Binary inst) {
  const auto *lhs = &inst.getLhsAsOperand();
  const auto *rhs = &inst.getRhsAsOperand();
  auto dest = inst.getResult();
  auto rd = translater.getRegister(dest);
  assert(!lhs->isConstant() ||
         !rhs->isConstant() &&
             "Binary XOR must have at least one non-constant operand. "
             "It is guaranteed by `OutlineConstant` pass.");

  if (lhs->isConstant() | rhs->isConstant()) {
    // use itype
    if (lhs->isConstant())
      std::swap(lhs, rhs);

    auto *rhsImm = getImmediate(rhs->getDefiningInst<ir::inst::Constant>());
    assert(rhsImm && "Expected a constant immediate for rhs operand");
    auto lhsReg = translater.getOperandRegister(lhs);
    if (rhsImm->isZero()) {
      // mv dest, lhs
      builder.create<as::pseudo::Mv>(rd, lhsReg);
      return utils::LogicalResult::success();
    }

    builder.create<as::itype::Xori>(rd, lhsReg, rhsImm);
  } else {
    // use rtype
    auto lhsReg = translater.getOperandRegister(lhs);
    auto rhsReg = translater.getOperandRegister(rhs);

    builder.create<as::rtype::Xor>(rd, lhsReg, rhsReg);
  }

  return utils::LogicalResult::success();
}

utils::LogicalResult translateShl(as::AsmBuilder &builder,
                                  FunctionTranslater &translater,
                                  ir::inst::Binary inst) {
  const auto *lhs = &inst.getLhsAsOperand();
  const auto *rhs = &inst.getRhsAsOperand();
  auto dest = inst.getResult();
  auto rd = translater.getRegister(dest);
  assert(!lhs->isConstant() && "Binary shift left lhs must not be constant. "
                               "It is guaranteed by `OutlineConstant` pass.");
  auto dataSize = getDataSize(lhs->getType());

  if (rhs->isConstant()) {
    // use itype
    auto *rhsImm = getImmediate(rhs->getDefiningInst<ir::inst::Constant>());
    assert(rhsImm && "Expected a constant immediate for rhs operand");
    auto lhsReg = translater.getOperandRegister(lhs);
    if (rhsImm->isZero()) {
      // mv dest, lhs
      builder.create<as::pseudo::Mv>(rd, lhsReg);
      return utils::LogicalResult::success();
    }

    builder.create<as::itype::Slli>(rd, lhsReg, rhsImm, dataSize);
  } else {
    // use rtype
    auto lhsReg = translater.getOperandRegister(lhs);
    auto rhsReg = translater.getOperandRegister(rhs);

    builder.create<as::rtype::Sll>(rd, lhsReg, rhsReg, dataSize);
  }
  shiftDataByType(builder, lhs->getType().cast<ir::IntT>(), rd);
  return utils::LogicalResult::success();
}

utils::LogicalResult translateShr(as::AsmBuilder &builder,
                                  FunctionTranslater &translater,
                                  ir::inst::Binary inst) {
  const auto *lhs = &inst.getLhsAsOperand();
  const auto *rhs = &inst.getRhsAsOperand();
  auto dest = inst.getResult();
  auto rd = translater.getRegister(dest);
  assert(!lhs->isConstant() && "Binary shift right lhs must not be constant. "
                               "It is guaranteed by `OutlineConstant` pass.");

  auto dataSize = getDataSize(lhs->getType());
  if (rhs->isConstant()) {
    // use itype
    auto *rhsImm = getImmediate(rhs->getDefiningInst<ir::inst::Constant>());
    assert(rhsImm && "Expected a constant immediate for rhs operand");
    auto lhsReg = translater.getOperandRegister(lhs);
    if (rhsImm->isZero()) {
      // mv dest, lhs
      builder.create<as::pseudo::Mv>(rd, lhsReg);
      return utils::LogicalResult::success();
    }

    if (lhs->getType().cast<ir::IntT>().isSigned()) {
      builder.create<as::itype::Srai>(rd, lhsReg, rhsImm, dataSize);
    } else {
      builder.create<as::itype::Srli>(rd, lhsReg, rhsImm, dataSize);
    }
  } else {
    // use rtype
    auto lhsReg = translater.getOperandRegister(lhs);
    auto rhsReg = translater.getOperandRegister(rhs);

    if (lhs->getType().cast<ir::IntT>().isSigned())
      builder.create<as::rtype::Sra>(rd, lhsReg, rhsReg, dataSize);
    else
      builder.create<as::rtype::Srl>(rd, lhsReg, rhsReg, dataSize);
  }

  return utils::LogicalResult::success();
}

utils::LogicalResult translateEq(as::AsmBuilder &builder,
                                 FunctionTranslater &translater,
                                 ir::inst::Binary inst) {
  const auto *lhs = &inst.getLhsAsOperand();
  const auto *rhs = &inst.getRhsAsOperand();
  auto dest = inst.getResult();
  auto rd = translater.getRegister(dest);
  auto dataSize = getDataSize(lhs->getType());

  if (lhs->getType().isa<ir::FloatT>()) {
    assert(!lhs->isConstant() && !rhs->isConstant() &&
           "Binary float equality operands must not be constant. "
           "It is guaranteed by `OutlineConstant` pass.");
    auto lhsReg = translater.getOperandRegister(lhs);
    auto rhsReg = translater.getOperandRegister(rhs);
    builder.create<as::rtype::Feq>(rd, lhsReg, rhsReg, dataSize);
  } else {
    assert(!lhs->isConstant() ||
           !rhs->isConstant() &&
               "Binary equality must have at least one non-constant operand. "
               "It is guaranteed by `OutlineConstant` pass.");

    if (lhs->isConstant() | rhs->isConstant()) {
      // use seqz
      if (lhs->isConstant())
        std::swap(lhs, rhs);

      auto lhsReg = translater.getOperandRegister(lhs);
      auto *rhsImm = getImmediate(rhs->getDefiningInst<ir::inst::Constant>());
      assert(rhsImm);
      if (rhsImm->isZero()) {
        builder.create<as::pseudo::Seqz>(rd, lhsReg);
        return utils::LogicalResult::success();
      } else {
        builder.create<as::itype::Xori>(rd, lhsReg, rhsImm);
      }
    } else {
      // xor rd, lhs, rhs
      builder.create<as::rtype::Xor>(rd, translater.getOperandRegister(lhs),
                                     translater.getOperandRegister(rhs));
    }

    // seqz rd, rd
    builder.create<as::pseudo::Seqz>(rd, rd);
  }

  return utils::LogicalResult::success();
}

utils::LogicalResult translateNeq(as::AsmBuilder &builder,
                                  FunctionTranslater &translater,
                                  ir::inst::Binary inst) {
  const auto *lhs = &inst.getLhsAsOperand();
  const auto *rhs = &inst.getRhsAsOperand();
  auto rd = translater.getRegister(inst);
  auto type = lhs->getType();

  if (type.isa<ir::FloatT>()) {
    assert(!lhs->isConstant() && !rhs->isConstant() &&
           "Binary float inequality operands must not be constant. "
           "It is guaranteed by `OutlineConstant` pass.");
    auto lhsReg = translater.getOperandRegister(lhs);
    auto rhsReg = translater.getOperandRegister(rhs);
    auto dataSize = getDataSize(type);
    builder.create<as::rtype::Feq>(rd, lhsReg, rhsReg, dataSize);
    createLogicalNot(builder, rd);
  } else {
    assert(!lhs->isConstant() ||
           !rhs->isConstant() &&
               "Binary equality must have at least one non-constant operand. "
               "It is guaranteed by `OutlineConstant` pass.");

    if (lhs->isConstant() | rhs->isConstant()) {
      // use seqz
      if (lhs->isConstant())
        std::swap(lhs, rhs);

      auto lhsReg = translater.getOperandRegister(lhs);
      auto *rhsImm = getImmediate(rhs->getDefiningInst<ir::inst::Constant>());
      assert(rhsImm);
      if (rhsImm->isZero()) {
        builder.create<as::pseudo::Seqz>(rd, lhsReg);
        return utils::LogicalResult::success();
      } else {
        builder.create<as::itype::Xori>(rd, lhsReg, rhsImm);
      }
    } else {
      // xor rd, lhs, rhs
      builder.create<as::rtype::Xor>(rd, translater.getOperandRegister(lhs),
                                     translater.getOperandRegister(rhs));
    }

    builder.create<as::rtype::Slt>(rd, as::Register::zero(), rd, false);
  }

  return utils::LogicalResult::success();
}

utils::LogicalResult translateLt(as::AsmBuilder &builder,
                                 FunctionTranslater &translater,
                                 ir::inst::Binary inst) {
  const auto *lhs = &inst.getLhsAsOperand();
  const auto *rhs = &inst.getRhsAsOperand();
  auto dest = inst.getResult();
  auto rd = translater.getRegister(dest);
  auto dataSize = getDataSize(lhs->getType());

  if (lhs->getType().isa<ir::FloatT>()) {
    assert(!lhs->isConstant() && !rhs->isConstant() &&
           "Binary float less than operands must not be constant. "
           "It is guaranteed by `OutlineConstant` pass.");
    auto lhsReg = translater.getOperandRegister(lhs);
    auto rhsReg = translater.getOperandRegister(rhs);
    builder.create<as::rtype::Flt>(rd, lhsReg, rhsReg, dataSize);
  } else {
    assert(!lhs->isConstant() && "Binary less than lhs must not be constant. "
                                 "It is guaranteed by `OutlineConstant` pass.");

    if (rhs->isConstant()) {
      // use itype
      auto *rhsImm = getImmediate(rhs->getDefiningInst<ir::inst::Constant>());
      assert(rhsImm && "Expected a constant immediate for rhs operand");
      auto lhsReg = translater.getOperandRegister(lhs);

      auto isSigned = rhs->getType().cast<ir::IntT>().isSigned();
      builder.create<as::itype::Slti>(rd, lhsReg, rhsImm, isSigned);
    } else {
      // use rtype
      auto lhsReg = translater.getOperandRegister(lhs);
      auto rhsReg = translater.getOperandRegister(rhs);

      auto isSigned = rhs->getType().cast<ir::IntT>().isSigned();
      builder.create<as::rtype::Slt>(rd, lhsReg, rhsReg, isSigned);
    }
  }
  return utils::LogicalResult::success();
}
utils::LogicalResult translateLe(as::AsmBuilder &builder,
                                 FunctionTranslater &translater,
                                 ir::inst::Binary inst) {
  // lt rd, rhs, lhs
  // xori rd, rd, 1

  const auto *lhs = &inst.getLhsAsOperand();
  const auto *rhs = &inst.getRhsAsOperand();
  auto dest = inst.getResult();
  auto rd = translater.getRegister(dest);
  auto dataSize = getDataSize(lhs->getType());

  if (lhs->getType().isa<ir::FloatT>()) {
    assert(!lhs->isConstant() && !rhs->isConstant() &&
           "Binary float less than or equal operands must not be constant. "
           "It is guaranteed by `OutlineConstant` pass.");
    auto lhsReg = translater.getOperandRegister(lhs);
    auto rhsReg = translater.getOperandRegister(rhs);
    builder.create<as::rtype::Fle>(rd, lhsReg, rhsReg, dataSize);
    return utils::LogicalResult::success();
  } else {
    assert(!rhs->isConstant() &&
           "Binary less than or equal rhs must not be constant. ");

    if (lhs->isConstant()) {
      // use itype
      auto *lhsImm = getImmediate(lhs->getDefiningInst<ir::inst::Constant>());
      assert(lhsImm && "Expected a constant immediate for lhs operand");
      auto rhsReg = translater.getOperandRegister(rhs);

      auto isSigned = lhs->getType().cast<ir::IntT>().isSigned();
      builder.create<as::itype::Slti>(rd, rhsReg, lhsImm, isSigned);
    } else {
      // use rtype
      auto lhsReg = translater.getOperandRegister(lhs);
      auto rhsReg = translater.getOperandRegister(rhs);

      auto isSigned = lhs->getType().cast<ir::IntT>().isSigned();
      builder.create<as::rtype::Slt>(rd, rhsReg, lhsReg, isSigned);
    }
  }

  // xori rd, rd, 1
  createLogicalNot(builder, rd);
  return utils::LogicalResult::success();
}
utils::LogicalResult translateGt(as::AsmBuilder &builder,
                                 FunctionTranslater &translater,
                                 ir::inst::Binary inst) {
  // lt rd, rhs, lhs

  const auto *lhs = &inst.getLhsAsOperand();
  const auto *rhs = &inst.getRhsAsOperand();
  auto dest = inst.getResult();
  auto rd = translater.getRegister(dest);
  auto dataSize = getDataSize(lhs->getType());

  if (lhs->getType().isa<ir::FloatT>()) {
    assert(!lhs->isConstant() && !rhs->isConstant() &&
           "Binary float greater than operands must not be constant. "
           "It is guaranteed by `OutlineConstant` pass.");
    auto lhsReg = translater.getOperandRegister(lhs);
    auto rhsReg = translater.getOperandRegister(rhs);
    builder.create<as::rtype::Flt>(rd, rhsReg, lhsReg, dataSize);
  } else {
    assert(!rhs->isConstant() &&
           "Binary greater than rhs must not be constant. ");

    if (lhs->isConstant()) {
      // use itype
      auto *lhsImm = getImmediate(lhs->getDefiningInst<ir::inst::Constant>());
      assert(lhsImm && "Expected a constant immediate for lhs operand");
      auto rhsReg = translater.getOperandRegister(rhs);

      auto isSigned = lhs->getType().cast<ir::IntT>().isSigned();
      builder.create<as::itype::Slti>(rd, rhsReg, lhsImm, isSigned);
    } else {
      // use rtype
      auto lhsReg = translater.getOperandRegister(lhs);
      auto rhsReg = translater.getOperandRegister(rhs);

      auto isSigned = lhs->getType().cast<ir::IntT>().isSigned();
      builder.create<as::rtype::Slt>(rd, rhsReg, lhsReg, isSigned);
    }
  }

  return utils::LogicalResult::success();
}
utils::LogicalResult translateGe(as::AsmBuilder &builder,
                                 FunctionTranslater &translater,
                                 ir::inst::Binary inst) {
  // lt rd, lhs, rhs
  // xori rd, rd, 1

  const auto *lhs = &inst.getLhsAsOperand();
  const auto *rhs = &inst.getRhsAsOperand();
  auto dest = inst.getResult();
  auto rd = translater.getRegister(dest);
  auto dataSize = getDataSize(lhs->getType());

  if (lhs->getType().isa<ir::FloatT>()) {
    assert(!lhs->isConstant() && !rhs->isConstant() &&
           "Binary float less than operands must not be constant. "
           "It is guaranteed by `OutlineConstant` pass.");
    auto lhsReg = translater.getOperandRegister(lhs);
    auto rhsReg = translater.getOperandRegister(rhs);
    builder.create<as::rtype::Fle>(rd, rhsReg, lhsReg, dataSize);
    return utils::LogicalResult::success();
  } else {
    assert(!lhs->isConstant() && "Binary less than lhs must not be constant. "
                                 "It is guaranteed by `OutlineConstant` pass.");

    if (rhs->isConstant()) {
      // use itype
      auto *rhsImm = getImmediate(rhs->getDefiningInst<ir::inst::Constant>());
      assert(rhsImm && "Expected a constant immediate for rhs operand");
      auto lhsReg = translater.getOperandRegister(lhs);

      auto isSigned = rhs->getType().cast<ir::IntT>().isSigned();
      builder.create<as::itype::Slti>(rd, lhsReg, rhsImm, isSigned);
    } else {
      // use rtype
      auto lhsReg = translater.getOperandRegister(lhs);
      auto rhsReg = translater.getOperandRegister(rhs);

      auto isSigned = rhs->getType().cast<ir::IntT>().isSigned();
      builder.create<as::rtype::Slt>(rd, lhsReg, rhsReg, isSigned);
    }
  }

  // xori rd, rd, 1
  createLogicalNot(builder, rd);
  return utils::LogicalResult::success();
}

} // namespace

//===----------------------------------------------------------------------===//
/// Unary Instruction Translation
//===----------------------------------------------------------------------===//

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
    if (operand->getType().isa<ir::FloatT>())
      builder.create<as::pseudo::Fneg>(dataSize, rd, rd);
    else
      // sub rd, zero, rs
      builder.create<as::rtype::Sub>(rd, as::Register::zero(), rs, dataSize);
  }
  if (operand->getType().isa<ir::IntT>())
    shiftDataByType(builder, operand->getType().cast<ir::IntT>(), rd);
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
  shiftDataByType(builder, operand->getType().cast<ir::IntT>(), rd);
  return utils::LogicalResult::success();
}

} // namespace

//===----------------------------------------------------------------------===//
/// Call Instruction Translation
//===----------------------------------------------------------------------===//

namespace {

static void callPreparation(as::AsmBuilder &builder,
                            FunctionTranslater &translater,
                            llvm::ArrayRef<ir::Operand> operands) {
  llvm::SmallVector<DataSpace, 8> operandIntRegs;
  llvm::SmallVector<DataSpace, 8> operandFloatRegs;

  for (const auto &operand : operands) {
    assert(!operand.isConstant() &&
           "Constant operands should be handled separately");
    auto reg = DataSpace::value(translater.getRegister(operand));
    if (auto spillMem = translater.getSpillData(operand))
      reg = *spillMem;

    if (operand.getType().isa<ir::FloatT>())
      operandFloatRegs.emplace_back(reg);
    else
      operandIntRegs.emplace_back(reg);
  }

  llvm::SmallVector<DataSpace, 8> intArgRegs;
  llvm::SmallVector<DataSpace, 8> floatArgRegs;

  size_t argMemorySize = 0;
  FunctionStack *stack = translater.getStack();
  for (auto idx = 0u; idx < operands.size(); ++idx) {
    auto type = operands[idx].getType();
    if (type.isa<ir::FloatT>()) {
      if (floatArgRegs.size() >= translater.getFloatArgRegisters().size()) {
        auto sp = stack->callArgument(argMemorySize);
        argMemorySize +=
            translater.getModule()->getContext()->getArchitectureByteSize();
        auto anonReg = translater.createAnonRegister(
            as::RegisterType::FloatingPoint, sp,
            as::DataSize::doublePrecision(), false);
        floatArgRegs.emplace_back(DataSpace::memory(anonReg));
      } else {
        floatArgRegs.emplace_back(DataSpace::value(
            translater.getFloatArgRegisters()[floatArgRegs.size()]));
      }

    } else {
      if (intArgRegs.size() >= translater.getIntArgRegisters().size()) {
        // argument memory
        auto sp = stack->callArgument(argMemorySize);
        argMemorySize +=
            translater.getModule()->getContext()->getArchitectureByteSize();
        auto anonReg = translater.createAnonRegister(
            as::RegisterType::Integer, sp, as::DataSize::doubleWord(), false);
        intArgRegs.emplace_back(DataSpace::memory(anonReg));
      } else {
        intArgRegs.emplace_back(DataSpace::value(
            translater.getIntArgRegisters()[intArgRegs.size()]));
      }
    }
  }

  stack->setCallArgumentsSize(
      std::max(argMemorySize, stack->getCallArgumentsSize()));

  translater.moveRegisters(builder, operandIntRegs, intArgRegs);
  translater.moveRegisters(builder, operandFloatRegs, floatArgRegs);
}

static void finalizeCall(as::AsmBuilder &builder,
                         FunctionTranslater &translater,
                         llvm::ArrayRef<ir::Type> retTypes,
                         llvm::ArrayRef<as::Register> resultRegs) {
  static llvm::SmallVector<as::Register> intRetRegs = {as::Register::a0(),
                                                       as::Register::a1()};
  static llvm::SmallVector<as::Register> floatRetRegs = {as::Register::fa0(),
                                                         as::Register::fa1()};

  llvm::SmallVector<DataSpace, 8> intFromCallee;
  llvm::SmallVector<DataSpace, 8> floatFromCallee;

  for (auto idx = 0u, intIdx = 0u, floatIdx = 0u; idx < retTypes.size();
       ++idx) {
    if (retTypes[idx].isa<ir::FloatT>())
      floatFromCallee.emplace_back(DataSpace::value(floatRetRegs[floatIdx++]));
    else
      intFromCallee.emplace_back(DataSpace::value(intRetRegs[intIdx++]));
  }

  llvm::SmallVector<DataSpace, 8> intResultRegs;
  llvm::SmallVector<DataSpace, 8> floatResultRegs;

  for (auto idx = 0u, intIdx = 0u, floatIdx = 0u; idx < resultRegs.size();
       ++idx) {
    if (resultRegs[idx].isFloatingPoint())
      floatResultRegs.emplace_back(DataSpace::value(resultRegs[idx]));
    else
      intResultRegs.emplace_back(DataSpace::value(resultRegs[idx]));
  }

  translater.moveRegisters(builder, intFromCallee, intResultRegs);
  translater.moveRegisters(builder, floatFromCallee, floatResultRegs);
}

} // namespace

class CallTranslationRule : public InstructionTranslationRule<ir::inst::Call> {
public:
  CallTranslationRule() {}

  bool restoreActively() const override { return true; }
  bool callFunction() const override {
    return true; // Caller-saved registers should be saved before call
  }

  utils::LogicalResult translate(as::AsmBuilder &builder,
                                 FunctionTranslater &translater,
                                 ir::inst::Call inst) override {
    const auto *callee = &inst.getFunctionAsOperand();
    assert(!callee->isConstant() &&
           "Call callee must not be constant. It is guaranteed by "
           "`InlineCall` pass.");

    as::Register reg = [&]() {
      if (translater.getSpillData(*callee))
        return translater.restoreOperand(builder, callee);
      else
        return translater.getRegister(*callee);
    }();

    as::Register calleeReg = reg;
    if (reg.isArg()) {
      auto intRegs = translater.getTranslateContext()->getRegistersForAllocate(
          as::RegisterType::Integer);
      for (auto intReg : intRegs) {
        if (intReg.isTemp()) {
          calleeReg = intReg;
          break;
        }
      }
    }
    assert(!calleeReg.isArg() &&
           "No available temporary register for callee address");

    builder.create<as::pseudo::Mv>(calleeReg, reg);
    callPreparation(builder, translater, inst.getArguments());

    builder.create<as::pseudo::Jalr>(calleeReg);

    if (inst->getResultSize() == 1 &&
        inst->getResult(0).getType().isa<ir::UnitT>()) {
      // void function
      return utils::LogicalResult::success();
    }

    llvm::SmallVector<as::Register, 4> retRegs;
    for (ir::Value value : inst->getResults()) {
      retRegs.emplace_back(translater.getRegister(value));
    }

    finalizeCall(builder, translater,
                 callee->getType()
                     .cast<ir::PointerT>()
                     .getPointeeType()
                     .cast<ir::FunctionT>()
                     .getReturnTypes(),
                 retRegs);

    return utils::LogicalResult::success();
  }
};

class InlineCallTranslationRule
    : public InstructionTranslationRule<ir::inst::InlineCall> {
public:
  InlineCallTranslationRule() {}

  bool restoreActively() const override { return true; }
  bool callFunction() const override { return true; }

  utils::LogicalResult translate(as::AsmBuilder &builder,
                                 FunctionTranslater &translater,
                                 ir::inst::InlineCall inst) override {
    llvm::StringRef callee = inst.getName();
    ir::FunctionT funcT = inst.getFunctionType();

    callPreparation(builder, translater, inst.getArguments());
    builder.create<as::pseudo::Call>(callee);

    if (inst->getResultSize() == 1 &&
        inst->getResult(0).getType().isa<ir::UnitT>()) {
      // void function
      return utils::LogicalResult::success();
    }

    llvm::SmallVector<as::Register, 4> retRegs;
    for (ir::Value value : inst->getResults()) {
      retRegs.emplace_back(translater.getRegister(value));
    }

    finalizeCall(builder, translater, funcT.getReturnTypes(), retRegs);
    return utils::LogicalResult::success();
  }
};

class MemcpyTranslationRule
    : public InstructionTranslationRule<ir::inst::MemCpy> {
public:
  MemcpyTranslationRule() {}

  bool restoreActively() const override { return true; }
  bool callFunction() const override { return true; }

  utils::LogicalResult translate(as::AsmBuilder &builder,
                                 FunctionTranslater &translater,
                                 ir::inst::MemCpy memCpy) override {
    callPreparation(builder, translater, memCpy->getOperands());
    builder.create<as::pseudo::Call>("memcpy");
    return utils::LogicalResult::success();
  }
};

//===----------------------------------------------------------------------===//
/// Exit Instruction Translation
//===----------------------------------------------------------------------===//

namespace {

static utils::LogicalResult translateFloat(as::AsmBuilder &builder,
                                           FunctionTranslater &translater,
                                           llvm::APFloat value,
                                           ir::FloatT floatT, as::Register rd);

static utils::LogicalResult translatePointer(as::AsmBuilder &builder,
                                             FunctionTranslater &translater,
                                             ir::ConstantAttr inst,
                                             as::Register rd);

} // namespace

class ConstantTranslationRule
    : public InstructionTranslationRule<ir::inst::OutlineConstant> {
public:
  ConstantTranslationRule() {}

  utils::LogicalResult translate(as::AsmBuilder &builder,
                                 FunctionTranslater &translater,
                                 ir::inst::OutlineConstant inst) override {
    auto type = inst.getType();
    auto constant = inst.getConstant().getDefiningInst<ir::inst::Constant>();
    auto rd = translater.getRegister(inst.getResult());

    if (constant.getValue().isa<ir::ConstantUndefAttr>()) {
      // do nothing for undef
      // just use whatever in rd
      return utils::LogicalResult::success();
    }

    if (type.isa<ir::PointerT>()) {
      return translatePointer(builder, translater, constant.getValue(), rd);
    } else if (type.isa<ir::FloatT>()) {
      auto floatConstant = constant.getValue().cast<ir::ConstantFloatAttr>();
      return translateFloat(builder, translater, floatConstant.getValue(),
                            type.cast<ir::FloatT>(), rd);
    } else if (type.isa<ir::IntT>()) {
      auto intConstant = constant.getValue().cast<ir::ConstantIntAttr>();
      loadInt(builder, translater, rd, intConstant.getValue(),
              type.cast<ir::IntT>());
      return utils::LogicalResult::success();
    } else {
      llvm_unreachable("Unsupported constant type for translation");
    }
  }
};

namespace {

utils::LogicalResult translateFloat(as::AsmBuilder &builder,
                                    FunctionTranslater &translater,
                                    llvm::APFloat value, ir::FloatT floatT,
                                    as::Register rd) {
  loadFloat(builder, translater, rd, value);
  return utils::LogicalResult::success();
}

utils::LogicalResult translatePointer(as::AsmBuilder &builder,
                                      FunctionTranslater &translater,
                                      ir::ConstantAttr constant,
                                      as::Register rd) {
  auto [label, dataSize] = translater.getConstantLabel(constant);
  builder.create<as::pseudo::La>(rd, label);
  return utils::LogicalResult::success();
}

} // namespace

//===----------------------------------------------------------------------===//
/// Constant Instruction Translation
//===----------------------------------------------------------------------===//

class JumpTranslationRule : public InstructionTranslationRule<ir::inst::Jump> {
public:
  JumpTranslationRule() {}

  utils::LogicalResult translate(as::AsmBuilder &builder,
                                 FunctionTranslater &translater,
                                 ir::inst::Jump inst) override {
    ir::JumpArg *jumpArg = inst.getJumpArg();
    ir::Block *block = jumpArg->getBlock();

    std::string blockName = FunctionTranslater::getBlockName(block);

    builder.create<as::pseudo::J>(blockName);
    return utils::LogicalResult::success();
  }
};

class BranchTranslationRule
    : public InstructionTranslationRule<ir::inst::Branch> {
public:
  BranchTranslationRule() {}

  utils::LogicalResult translate(as::AsmBuilder &builder,
                                 FunctionTranslater &translater,
                                 ir::inst::Branch inst) override {
    ir::Block *thenArg = inst.getIfArg()->getBlock();
    ir::Block *elseArg = inst.getElseArg()->getBlock();

    const auto *condition = &inst.getConditionAsOperand();
    as::Register conditionReg = translater.getOperandRegister(condition);

    builder.create<as::btype::Beq>(conditionReg, as::Register::zero(),
                                   FunctionTranslater::getBlockName(elseArg));
    builder.create<as::pseudo::J>(FunctionTranslater::getBlockName(thenArg));
    return utils::LogicalResult::success();
  }
};

class SwitchTranslationRule
    : public InstructionTranslationRule<ir::inst::Switch> {
public:
  SwitchTranslationRule() {}

  utils::LogicalResult translate(as::AsmBuilder &builder,
                                 FunctionTranslater &translater,
                                 ir::inst::Switch inst) override {
    const auto *condition = &inst.getValueAsOperand();

    as::Register conditionReg = translater.getOperandRegister(condition);

    as::Register tempReg =
        translater.getTranslateContext()->getTempRegisters()[0];

    ir::Block *defaultBlock = inst.getDefaultCase()->getBlock();

    for (auto idx = 0u; idx < inst.getCaseSize(); ++idx) {
      ir::Block *caseBlock = inst.getCaseJumpArg(idx)->getBlock();
      auto caseValue = inst.getCaseValue(idx);
      assert(caseValue.isConstant());

      auto caseValueInt = caseValue.getDefiningInst<ir::inst::Constant>()
                              .getValue()
                              .cast<ir::ConstantIntAttr>();

      loadInt(builder, translater, tempReg, caseValueInt.getValue(),
              caseValueInt.getIntType());
      builder.create<as::btype::Beq>(
          conditionReg, tempReg, FunctionTranslater::getBlockName(caseBlock));
    }
    builder.create<as::pseudo::J>(
        FunctionTranslater::getBlockName(defaultBlock));
    return utils::LogicalResult::success();
  }
};

class ReturnTranslationRule
    : public InstructionTranslationRule<ir::inst::Return> {
public:
  ReturnTranslationRule() {}

  utils::LogicalResult translate(as::AsmBuilder &builder,
                                 FunctionTranslater &translater,
                                 ir::inst::Return inst) override {

    static llvm::SmallVector<DataSpace> intReturnRegisters{
        DataSpace::value(as::Register::a0()),
        DataSpace::value(as::Register::a1()),
    };

    static llvm::SmallVector<DataSpace> floatReturnRegisters = {
        DataSpace::value(as::Register::fa0()),
        DataSpace::value(as::Register::fa1()),
    };

    if (inst.getValueSize() == 1 &&
        inst.getValue(0).getType().isa<ir::UnitT>()) {
      // void return
    } else {
      llvm::SmallVector<DataSpace> intValueRegisters;
      llvm::SmallVector<DataSpace> floatValueRegisters;
      intValueRegisters.reserve(intReturnRegisters.size());
      floatValueRegisters.reserve(floatReturnRegisters.size());

      for (auto idx = 0u; idx < inst.getValueSize(); ++idx) {
        const auto *value = &inst.getValueAsOperand(idx);
        as::Register valueReg = translater.getOperandRegister(value);
        if (value->getType().isa<ir::FloatT>())
          floatValueRegisters.emplace_back(DataSpace::value(valueReg));
        else
          intValueRegisters.emplace_back(DataSpace::value(valueReg));
      }

      assert(intValueRegisters.size() <= intReturnRegisters.size() &&
             "Too many integer return values for available registers");
      assert(floatValueRegisters.size() <= floatReturnRegisters.size() &&
             "Too many floating point return values for available registers");

      if (!intValueRegisters.empty()) {
        translater.moveRegisters(builder, intValueRegisters,
                                 llvm::ArrayRef<DataSpace>(intReturnRegisters)
                                     .take_front(intValueRegisters.size()));
      }

      if (!floatValueRegisters.empty()) {
        translater.moveRegisters(builder, floatValueRegisters,
                                 llvm::ArrayRef<DataSpace>(floatReturnRegisters)
                                     .take_front(floatValueRegisters.size()));
      }
    }

    translater.writeFunctionEnd(builder);
    return utils::LogicalResult::success();
  }
};

class UnreachableTranslationRule
    : public InstructionTranslationRule<ir::inst::Unreachable> {
public:
  UnreachableTranslationRule() {}

  utils::LogicalResult translate(as::AsmBuilder &builder,
                                 FunctionTranslater &translater,
                                 ir::inst::Unreachable inst) override {
    builder.create<as::pseudo::Call>(KECC_UNREACHABLE_LABEL);
    return utils::LogicalResult::success();
  }
};
//===----------------------------------------------------------------------===//
/// Memory Instruction Translation
//===----------------------------------------------------------------------===//

class StoreTranslationRule
    : public InstructionTranslationRule<ir::inst::Store> {
public:
  StoreTranslationRule() {}

  utils::LogicalResult translate(as::AsmBuilder &builder,
                                 FunctionTranslater &translater,
                                 ir::inst::Store inst) override {
    const auto *value = &inst.getValueAsOperand();
    const auto *ptr = &inst.getPointerAsOperand();
    auto valueType = value->getType();
    auto dataSize = getDataSize(valueType);

    auto valueReg = translater.getOperandRegister(value);
    auto ptrReg = translater.getOperandRegister(ptr);

    builder.create<as::stype::Store>(ptrReg, valueReg, getImmediate(0),
                                     dataSize);
    return utils::LogicalResult::success();
  }
};

class LoadTranslationRule : public InstructionTranslationRule<ir::inst::Load> {
public:
  LoadTranslationRule() {}

  utils::LogicalResult translate(as::AsmBuilder &builder,
                                 FunctionTranslater &translater,
                                 ir::inst::Load inst) override {
    const auto *ptr = &inst.getPointerAsOperand();
    auto dataSize = getDataSize(inst.getType());
    auto ptrReg = translater.getOperandRegister(ptr);
    auto rd = translater.getRegister(inst.getResult());

    bool isSigned = true;
    if (auto intT = inst.getType().dyn_cast<ir::IntT>()) {
      isSigned = intT.isSigned();
    }

    builder.create<as::itype::Load>(rd, ptrReg, getImmediate(0), dataSize,
                                    isSigned);
    return utils::LogicalResult::success();
  }
};

class GepTranslationRule : public InstructionTranslationRule<ir::inst::Gep> {
public:
  GepTranslationRule() {}

  utils::LogicalResult translate(as::AsmBuilder &builder,
                                 FunctionTranslater &translater,
                                 ir::inst::Gep inst) override {
    const auto *base = &inst.getBasePointerAsOperand();
    const auto *offset = &inst.getOffsetAsOperand();
    as::Register rd = translater.getRegister(inst.getResult());
    as::Register baseReg = translater.getOperandRegister(base);

    if (offset->isConstant()) {
      as::Immediate *offsetImm =
          getImmediate(offset->getDefiningInst<ir::inst::Constant>());

      builder.create<as::itype::Addi>(rd, baseReg, offsetImm,
                                      as::DataSize::doubleWord());
    } else {
      auto offsetReg = translater.getOperandRegister(offset);
      builder.create<as::rtype::Add>(rd, baseReg, offsetReg,
                                     as::DataSize::doubleWord());
    }

    return utils::LogicalResult::success();
  }
};

//===----------------------------------------------------------------------===//
/// Type Cast Instruction Translation
//===----------------------------------------------------------------------===//

static void castIntToInt(as::AsmBuilder &builder, as::Register rd,
                         as::Register srcReg, ir::IntT fromT, ir::IntT toT,
                         int archBitSize) {
  unsigned fromBitWidth = fromT.getBitWidth();
  unsigned toBitWidth = toT.getBitWidth();
  bool fromSigned = fromT.isSigned();
  bool toSigned = toT.isSigned();
  // lower bit int -> higher bit unsigned int : mv
  // lower bit signed int -> higher bit signed int : mv
  // lower bit unsigned int -> higher bit signed int : fill 0 in
  //                                                   higher bits
  // higher bit int -> lower unsigned bit : trucate higher bits
  // higher bit int -> lower signed bit : fill signed bit in higher bits

  if (fromBitWidth == 1)
    return;

  if (fromBitWidth == toBitWidth) {
    // mv rd, srcReg
    if (rd != srcReg)
      builder.create<as::pseudo::Mv>(rd, srcReg);
  } else if (fromBitWidth < toBitWidth) {
    if (!fromSigned && toSigned) {
      // fill 0 in higher bits
      auto bitDiff = archBitSize - fromBitWidth;
      builder.create<as::itype::Slli>(rd, srcReg, getImmediate(bitDiff),
                                      as::DataSize::doubleWord());
      builder.create<as::itype::Srli>(rd, rd, getImmediate(bitDiff),
                                      as::DataSize::doubleWord());
    } else if (rd != srcReg) {
      // mv rd, srcReg
      builder.create<as::pseudo::Mv>(rd, srcReg);
    }
  } else {
    auto bitDiff = archBitSize - toBitWidth;
    // shift left
    builder.create<as::itype::Slli>(rd, srcReg, getImmediate(bitDiff),
                                    as::DataSize::doubleWord());
    if (toSigned) {
      // srai
      builder.create<as::itype::Srai>(rd, rd, getImmediate(bitDiff),
                                      as::DataSize::doubleWord());
    } else {
      // srli
      builder.create<as::itype::Srli>(rd, rd, getImmediate(bitDiff),
                                      as::DataSize::doubleWord());
    }
  }
}

class TypeCastTranslationRule
    : public InstructionTranslationRule<ir::inst::TypeCast> {
public:
  TypeCastTranslationRule() {}

  utils::LogicalResult translate(as::AsmBuilder &builder,
                                 FunctionTranslater &translater,
                                 ir::inst::TypeCast inst) override {
    const auto *src = &inst.getValueAsOperand();
    auto srcType = src->getType();
    auto dstType = inst.getType();
    assert(ir::isCastableTo(srcType, dstType));

    auto rd = translater.getRegister(inst.getResult());
    auto srcReg = translater.getOperandRegister(src);

    auto fromDataSize = getDataSize(srcType);
    auto toDataSize = getDataSize(dstType);
    auto archBitSize =
        translater.getModule()->getContext()->getArchitectureBitSize();

    if (srcType.isa<ir::PointerT>()) {
      if (dstType.isa<ir::IntT>()) {
        castIntToInt(builder, rd, srcReg,
                     ir::IntT::get(srcType.getContext(), archBitSize, false),
                     dstType.cast<ir::IntT>(), archBitSize);
        return utils::LogicalResult::success();
      } else if (dstType.isa<ir::PointerT>()) {
        // Pointer to Pointer cast, just move the register
        builder.create<as::pseudo::Mv>(rd, srcReg);
        return utils::LogicalResult::success();
      }

      llvm_unreachable("Unsupported pointer cast type");
    } else if (auto srcIntT = srcType.dyn_cast<ir::IntT>()) {
      if (dstType.isa<ir::PointerT>()) {
        castIntToInt(builder, rd, srcReg, srcIntT,
                     ir::IntT::get(srcType.getContext(), archBitSize, false),
                     archBitSize);
        return utils::LogicalResult::success();
      } else if (auto dstIntT = dstType.dyn_cast<ir::IntT>()) {
        castIntToInt(builder, rd, srcReg, srcIntT, dstIntT, archBitSize);
      } else if (auto dstFloatT = dstType.dyn_cast<ir::FloatT>()) {
        // FcvIntToFloat
        builder.create<as::rtype::FcvtIntToFloat>(rd, srcReg, std::nullopt,
                                                  fromDataSize, toDataSize,
                                                  srcIntT.isSigned());
        return utils::LogicalResult::success();
      } else {
        llvm_unreachable("Unsupported int cast type");
      }
    } else if (auto srcFloatT = srcType.dyn_cast<ir::FloatT>()) {
      if (auto dstIntT = dstType.dyn_cast<ir::IntT>()) {
        // FcvFloatToInt
        builder.create<as::rtype::FcvtFloatToInt>(rd, srcReg, std::nullopt,
                                                  fromDataSize, toDataSize,
                                                  dstIntT.isSigned());
        return utils::LogicalResult::success();
      } else if (dstType.isa<ir::FloatT>()) {
        // FcvFloatToFloat
        builder.create<as::rtype::FcvtFloatToFloat>(rd, srcReg, std::nullopt,
                                                    fromDataSize, toDataSize);
        return utils::LogicalResult::success();
      } else {
        llvm_unreachable("Unsupported float cast type");
      }
    } else {
      llvm_unreachable("Unsupported type cast");
    }

    return utils::LogicalResult::success();
  }
};

class NopTranslationRule : public InstructionTranslationRule<ir::inst::Nop> {
public:
  NopTranslationRule() {}

  utils::LogicalResult translate(as::AsmBuilder &builder,
                                 FunctionTranslater &translater,
                                 ir::inst::Nop inst) override {
    // Do nothing for Nop
    return utils::LogicalResult::success();
  }
};

class FunctionArgTranslationRule
    : public InstructionTranslationRule<ir::inst::FunctionArgument> {
public:
  FunctionArgTranslationRule() {}

  bool spillActively() const override { return true; }

  utils::LogicalResult translate(as::AsmBuilder &builder,
                                 FunctionTranslater &translater,
                                 ir::inst::FunctionArgument inst) override {
    // Function arguments are handled in the prologue of the function.
    return utils::LogicalResult::success();
  }
};

class CopyTranslationRule
    : public InstructionTranslationRule<translate::inst::Copy> {
public:
  CopyTranslationRule() {}

  utils::LogicalResult translate(as::AsmBuilder &builder,
                                 FunctionTranslater &translater,
                                 translate::inst::Copy copy) override {
    auto src = translater.getOperandRegister(&copy.getValueAsOperand());
    auto dst = translater.getRegister(copy);

    if (src != dst) {
      builder.create<as::pseudo::Mv>(dst, src);
    }
    return utils::LogicalResult::success();
  }
};

class LoadOffsetTranslationRule
    : public InstructionTranslationRule<translate::inst::LoadOffset> {
public:
  LoadOffsetTranslationRule() {}

  utils::LogicalResult
  translate(as::AsmBuilder &builder, FunctionTranslater &translater,
            translate::inst::LoadOffset loadOffset) override {
    const auto *ptr = &loadOffset.getPointerAsOperand();
    auto dataSize = getDataSize(loadOffset.getType());
    auto ptrReg = translater.getOperandRegister(ptr);
    auto rd = translater.getRegister(loadOffset.getResult());
    auto offsetImm = getImmediate(loadOffset.getOffset());
    assert(offsetImm && "Expected a constant immediate for offset");
    bool isSigned = true;
    if (auto intT = loadOffset.getType().dyn_cast<ir::IntT>()) {
      isSigned = intT.isSigned();
    }
    builder.create<as::itype::Load>(rd, ptrReg, offsetImm, dataSize, isSigned);
    return utils::LogicalResult::success();
  }
};

class StoreOffsetTranslationRule
    : public InstructionTranslationRule<translate::inst::StoreOffset> {
public:
  StoreOffsetTranslationRule() {}

  utils::LogicalResult
  translate(as::AsmBuilder &builder, FunctionTranslater &translater,
            translate::inst::StoreOffset storeOffset) override {
    const auto *value = &storeOffset.getValueAsOperand();
    const auto *ptr = &storeOffset.getPointerAsOperand();
    auto valueType = value->getType();
    auto dataSize = getDataSize(valueType);
    auto valueReg = translater.getOperandRegister(value);
    auto ptrReg = translater.getOperandRegister(ptr);
    auto offsetImm = getImmediate(storeOffset.getOffset());
    assert(offsetImm && "Expected a constant immediate for offset");
    builder.create<as::stype::Store>(ptrReg, valueReg, offsetImm, dataSize);
    return utils::LogicalResult::success();
  }
};

void registerDefaultTranslationRules(TranslateContext *context) {
  registerTranslationRule<BinaryTranslationRule>(context);
  registerTranslationRule<UnaryTranslationRule>(context);
  registerTranslationRule<ConstantTranslationRule>(context);
  registerTranslationRule<JumpTranslationRule>(context);
  registerTranslationRule<BranchTranslationRule>(context);
  registerTranslationRule<SwitchTranslationRule>(context);
  registerTranslationRule<ReturnTranslationRule>(context);
  registerTranslationRule<UnreachableTranslationRule>(context);
  registerTranslationRule<StoreTranslationRule>(context);
  registerTranslationRule<LoadTranslationRule>(context);
  registerTranslationRule<GepTranslationRule>(context);
  registerTranslationRule<TypeCastTranslationRule>(context);
  registerTranslationRule<NopTranslationRule>(context);
  registerTranslationRule<CallTranslationRule>(context);
  registerTranslationRule<InlineCallTranslationRule>(context);
  registerTranslationRule<MemcpyTranslationRule>(context);
  registerTranslationRule<FunctionArgTranslationRule>(context);
  registerTranslationRule<CopyTranslationRule>(context);
  registerTranslationRule<LoadOffsetTranslationRule>(context);
  registerTranslationRule<StoreOffsetTranslationRule>(context);
}

} // namespace kecc
