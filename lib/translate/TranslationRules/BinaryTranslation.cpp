#include "kecc/asm/Asm.h"
#include "kecc/asm/AsmInstruction.h"
#include "kecc/ir/IRInstructions.h"
#include "kecc/ir/IRTypes.h"
#include "kecc/translate/IRTranslater.h"
#include "kecc/utils/LogicalResult.h"

namespace kecc {

namespace {
static utils::LogicalResult translateAdd(as::AsmBuilder &builder,
                                         FunctionTranslater &translater,
                                         ir::inst::Binary inst);
static utils::LogicalResult translateIntSub(as::AsmBuilder &builder,
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
    as::AsmBuilder &, FunctionTranslater &, ir::inst::Binary) = {
    translateAdd, translateIntSub, translateMul, translateDiv,
    translateMod, translateAnd,    translateOr,  translateXor,
    translateShl, translateShr,    translateEq,  translateNeq,
    translateLt,  translateLe,     translateGt,  translateGe};

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

utils::LogicalResult translateIntAdd(as::AsmBuilder &builder,
                                     FunctionTranslater &translater,
                                     ir::inst::Binary inst) {
  auto &regAlloc = translater.regAlloc();
  auto lhs = inst.getLhs();
  auto rhs = inst.getRhs();
  auto dest = inst.getResult();
  auto rd = regAlloc.getRegister(dest);
  auto dataSize = getDataSize(lhs.getType());

  assert(!lhs.isConstant() ||
         !rhs.isConstant() &&
             "Binary addition must have at least one non-constant operand. "
             "It is guaranteed by `OutlineConstant` pass.");

  if (lhs.isConstant() | rhs.isConstant()) {
    // use itype
    if (lhs.isConstant())
      std::swap(lhs, rhs);

    auto *rhsImm = getImmediate(rhs.getDefiningInst<ir::inst::Constant>());
    assert(rhsImm && "Expected a constant immediate for rhs operand");

    auto lhsReg = regAlloc.getRegister(lhs);
    if (rhsImm->isZero()) {
      // mv dest, lhs
      builder.create<as::pseudo::Mv>(rd, lhsReg);
      return utils::LogicalResult::success();
    }

    builder.create<as::itype::Addi>(rd, lhsReg, rhsImm, dataSize);
    return utils::LogicalResult::success();
  } else {
    // use rtype

    auto lhsReg = regAlloc.getRegister(lhs);
    auto rhsReg = regAlloc.getRegister(rhs);

    builder.create<as::rtype::Add>(rd, lhsReg, rhsReg, dataSize);
    return utils::LogicalResult::success();
  }
}

utils::LogicalResult translateFloatAdd(as::AsmBuilder &builder,
                                       FunctionTranslater &translater,
                                       ir::inst::Binary inst) {
  auto &regAlloc = translater.regAlloc();
  auto lhs = inst.getLhs();
  auto rhs = inst.getRhs();
  auto dest = inst.getResult();
  auto rd = regAlloc.getRegister(dest);
  auto dataSize = getDataSize(lhs.getType());

  assert(!lhs.isConstant() && !rhs.isConstant() &&
         "Binary float addition operands must not be constant. "
         "It is guaranteed by `OutlineConstant` pass.");

  auto lhsReg = regAlloc.getRegister(lhs);
  auto rhsReg = regAlloc.getRegister(rhs);
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
  auto &regAlloc = translater.regAlloc();
  auto lhs = inst.getLhs();
  auto rhs = inst.getRhs();
  auto dest = inst.getResult();
  auto rd = regAlloc.getRegister(dest);
  auto dataSize = getDataSize(lhs.getType());

  assert(!lhs.isConstant() && "Binary subtraction lhs must not be constant. "
                              "It is guaranteed by `OutlineConstant` pass.");
  auto lhsReg = regAlloc.getRegister(lhs);

  if (rhs.isConstant()) {
    // use itype
    auto *rhsImm = getImmediate(rhs.getDefiningInst<ir::inst::Constant>());
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
    auto rhsReg = regAlloc.getRegister(rhs);
    builder.create<as::rtype::Sub>(rd, lhsReg, rhsReg, dataSize);
  }

  return utils::LogicalResult::success();
}

utils::LogicalResult translateFloatSub(as::AsmBuilder &builder,
                                       FunctionTranslater &translater,
                                       ir::inst::Binary inst) {
  auto &regAlloc = translater.regAlloc();
  auto lhs = inst.getLhs();
  auto rhs = inst.getRhs();
  auto dest = inst.getResult();
  auto rd = regAlloc.getRegister(dest);
  auto dataSize = getDataSize(lhs.getType());
  assert(!lhs.isConstant() && !rhs.isConstant() &&
         "Binary float subtraction operands must not be constant. "
         "It is guaranteed by `OutlineConstant` pass.");
  auto lhsReg = regAlloc.getRegister(lhs);
  auto rhsReg = regAlloc.getRegister(rhs);
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
  auto &regAlloc = translater.regAlloc();
  auto lhs = inst.getLhs();
  auto rhs = inst.getRhs();
  auto dest = inst.getResult();
  auto rd = regAlloc.getRegister(dest);
  auto dataSize = getDataSize(lhs.getType());
  assert(!lhs.isConstant() && !rhs.isConstant() &&
         "Binary multiplication operands must not be constant. "
         "It is guaranteed by `OutlineConstant` pass.");

  auto lhsReg = regAlloc.getRegister(lhs);
  auto rhsReg = regAlloc.getRegister(rhs);

  if (lhs.getType().isa<ir::FloatT>())
    builder.create<as::rtype::Fmul>(rd, lhsReg, rhsReg, dataSize);
  else
    builder.create<as::rtype::Mul>(rd, lhsReg, rhsReg, dataSize);
  return utils::LogicalResult::success();
}

utils::LogicalResult translateDiv(as::AsmBuilder &builder,
                                  FunctionTranslater &translater,
                                  ir::inst::Binary inst) {
  auto &regAlloc = translater.regAlloc();
  auto lhs = inst.getLhs();
  auto rhs = inst.getRhs();
  auto dest = inst.getResult();
  auto rd = regAlloc.getRegister(dest);
  auto dataSize = getDataSize(lhs.getType());
  assert(!lhs.isConstant() && !rhs.isConstant() &&
         "Binary division operands must not be constant. "
         "It is guaranteed by `OutlineConstant` pass.");

  auto lhsReg = regAlloc.getRegister(lhs);
  auto rhsReg = regAlloc.getRegister(rhs);

  if (lhs.getType().isa<ir::FloatT>())
    builder.create<as::rtype::Fdiv>(rd, lhsReg, rhsReg, dataSize);
  else {
    auto intT = rhs.getType().cast<ir::IntT>();
    builder.create<as::rtype::Div>(rd, lhsReg, rhsReg, dataSize,
                                   intT.isSigned());
  }

  return utils::LogicalResult::success();
}

utils::LogicalResult translateMod(as::AsmBuilder &builder,
                                  FunctionTranslater &translater,
                                  ir::inst::Binary inst) {
  auto &regAlloc = translater.regAlloc();
  auto lhs = inst.getLhs();
  auto rhs = inst.getRhs();
  auto dest = inst.getResult();
  auto rd = regAlloc.getRegister(dest);
  auto dataSize = getDataSize(lhs.getType());
  assert(!lhs.isConstant() && !rhs.isConstant() &&
         "Binary modulo operands must not be constant. "
         "It is guaranteed by `OutlineConstant` pass.");
  assert(!lhs.getType().isa<ir::FloatT>() &&
         "Binary modulo operands must not be float type. ");
  auto lhsReg = regAlloc.getRegister(lhs);
  auto rhsReg = regAlloc.getRegister(rhs);

  auto intT = rhs.getType().cast<ir::IntT>();

  builder.create<as::rtype::Rem>(rd, lhsReg, rhsReg, dataSize, intT.isSigned());
  return utils::LogicalResult::success();
}

utils::LogicalResult translateAnd(as::AsmBuilder &builder,
                                  FunctionTranslater &translater,
                                  ir::inst::Binary inst) {
  auto &regAlloc = translater.regAlloc();
  auto lhs = inst.getLhs();
  auto rhs = inst.getRhs();
  auto dest = inst.getResult();
  auto rd = regAlloc.getRegister(dest);

  assert(!lhs.isConstant() ||
         !rhs.isConstant() &&
             "Binary AND must have at least one non-constant operand. "
             "It is guaranteed by `OutlineConstant` pass.");

  if (lhs.isConstant() | rhs.isConstant()) {
    // use itype
    if (lhs.isConstant())
      std::swap(lhs, rhs);

    auto *rhsImm = getImmediate(rhs.getDefiningInst<ir::inst::Constant>());
    assert(rhsImm && "Expected a constant immediate for rhs operand");
    auto lhsReg = regAlloc.getRegister(lhs);
    if (rhsImm->isZero()) {
      // mv dest, zero
      builder.create<as::pseudo::Mv>(rd, as::Register::zero());
      return utils::LogicalResult::success();
    }

    builder.create<as::itype::Andi>(rd, lhsReg, rhsImm);
    return utils::LogicalResult::success();
  } else {
    // use rtype
    auto lhsReg = regAlloc.getRegister(lhs);
    auto rhsReg = regAlloc.getRegister(rhs);

    builder.create<as::rtype::And>(rd, lhsReg, rhsReg);
    return utils::LogicalResult::success();
  }
}

utils::LogicalResult translateOr(as::AsmBuilder &builder,
                                 FunctionTranslater &translater,
                                 ir::inst::Binary inst) {
  auto &regAlloc = translater.regAlloc();
  auto lhs = inst.getLhs();
  auto rhs = inst.getRhs();
  auto dest = inst.getResult();
  auto rd = regAlloc.getRegister(dest);
  assert(!lhs.isConstant() ||
         !rhs.isConstant() &&
             "Binary OR must have at least one non-constant operand. "
             "It is guaranteed by `OutlineConstant` pass.");

  if (lhs.isConstant() | rhs.isConstant()) {
    if (lhs.isConstant())
      std::swap(lhs, rhs);

    // use itype
    auto *rhsImm = getImmediate(rhs.getDefiningInst<ir::inst::Constant>());
    assert(rhsImm && "Expected a constant immediate for rhs operand");
    auto lhsReg = regAlloc.getRegister(lhs);
    if (rhsImm->isZero()) {
      // mv dest, lhs
      builder.create<as::pseudo::Mv>(rd, lhsReg);
      return utils::LogicalResult::success();
    }

    builder.create<as::itype::Ori>(rd, lhsReg, rhsImm);
    return utils::LogicalResult::success();
  } else {
    // use rtype
    auto lhsReg = regAlloc.getRegister(lhs);
    auto rhsReg = regAlloc.getRegister(rhs);

    builder.create<as::rtype::Or>(rd, lhsReg, rhsReg);
    return utils::LogicalResult::success();
  }
}

utils::LogicalResult translateXor(as::AsmBuilder &builder,
                                  FunctionTranslater &translater,
                                  ir::inst::Binary inst) {
  auto &regAlloc = translater.regAlloc();
  auto lhs = inst.getLhs();
  auto rhs = inst.getRhs();
  auto dest = inst.getResult();
  auto rd = regAlloc.getRegister(dest);
  assert(!lhs.isConstant() ||
         !rhs.isConstant() &&
             "Binary XOR must have at least one non-constant operand. "
             "It is guaranteed by `OutlineConstant` pass.");

  if (lhs.isConstant() | rhs.isConstant()) {
    // use itype
    if (lhs.isConstant())
      std::swap(lhs, rhs);

    auto *rhsImm = getImmediate(rhs.getDefiningInst<ir::inst::Constant>());
    assert(rhsImm && "Expected a constant immediate for rhs operand");
    auto lhsReg = regAlloc.getRegister(lhs);
    if (rhsImm->isZero()) {
      // mv dest, lhs
      builder.create<as::pseudo::Mv>(rd, lhsReg);
      return utils::LogicalResult::success();
    }

    builder.create<as::itype::Xori>(rd, lhsReg, rhsImm);
  } else {
    // use rtype
    auto lhsReg = regAlloc.getRegister(lhs);
    auto rhsReg = regAlloc.getRegister(rhs);

    builder.create<as::rtype::Xor>(rd, lhsReg, rhsReg);
  }

  return utils::LogicalResult::success();
}

utils::LogicalResult translateShl(as::AsmBuilder &builder,
                                  FunctionTranslater &translater,
                                  ir::inst::Binary inst) {
  auto &regAlloc = translater.regAlloc();
  auto lhs = inst.getLhs();
  auto rhs = inst.getRhs();
  auto dest = inst.getResult();
  auto rd = regAlloc.getRegister(dest);
  assert(!lhs.isConstant() && "Binary shift left lhs must not be constant. "
                              "It is guaranteed by `OutlineConstant` pass.");
  auto dataSize = getDataSize(lhs.getType());

  if (rhs.isConstant()) {
    // use itype
    auto *rhsImm = getImmediate(rhs.getDefiningInst<ir::inst::Constant>());
    assert(rhsImm && "Expected a constant immediate for rhs operand");
    auto lhsReg = regAlloc.getRegister(lhs);
    if (rhsImm->isZero()) {
      // mv dest, lhs
      builder.create<as::pseudo::Mv>(rd, lhsReg);
      return utils::LogicalResult::success();
    }

    builder.create<as::itype::Slli>(rd, lhsReg, rhsImm, dataSize);
  } else {
    // use rtype
    auto lhsReg = regAlloc.getRegister(lhs);
    auto rhsReg = regAlloc.getRegister(rhs);

    builder.create<as::rtype::Sll>(rd, lhsReg, rhsReg, dataSize);
  }
  return utils::LogicalResult::success();
}

utils::LogicalResult translateShr(as::AsmBuilder &builder,
                                  FunctionTranslater &translater,
                                  ir::inst::Binary inst) {
  auto &regAlloc = translater.regAlloc();
  auto lhs = inst.getLhs();
  auto rhs = inst.getRhs();
  auto dest = inst.getResult();
  auto rd = regAlloc.getRegister(dest);
  assert(!lhs.isConstant() && "Binary shift right lhs must not be constant. "
                              "It is guaranteed by `OutlineConstant` pass.");

  auto dataSize = getDataSize(lhs.getType());
  if (rhs.isConstant()) {
    // use itype
    auto *rhsImm = getImmediate(rhs.getDefiningInst<ir::inst::Constant>());
    assert(rhsImm && "Expected a constant immediate for rhs operand");
    auto lhsReg = regAlloc.getRegister(lhs);
    if (rhsImm->isZero()) {
      // mv dest, lhs
      builder.create<as::pseudo::Mv>(rd, lhsReg);
      return utils::LogicalResult::success();
    }

    builder.create<as::itype::Srai>(rd, lhsReg, rhsImm, dataSize);
  } else {
    // use rtype
    auto lhsReg = regAlloc.getRegister(lhs);
    auto rhsReg = regAlloc.getRegister(rhs);

    builder.create<as::rtype::Sra>(rd, lhsReg, rhsReg, dataSize);
  }

  return utils::LogicalResult::success();
}

utils::LogicalResult translateEq(as::AsmBuilder &builder,
                                 FunctionTranslater &translater,
                                 ir::inst::Binary inst) {
  auto &regAlloc = translater.regAlloc();
  auto lhs = inst.getLhs();
  auto rhs = inst.getRhs();
  auto dest = inst.getResult();
  auto rd = regAlloc.getRegister(dest);
  auto dataSize = getDataSize(lhs.getType());

  if (lhs.getType().isa<ir::FloatT>()) {
    assert(!lhs.isConstant() && !rhs.isConstant() &&
           "Binary float equality operands must not be constant. "
           "It is guaranteed by `OutlineConstant` pass.");
    auto lhsReg = regAlloc.getRegister(lhs);
    auto rhsReg = regAlloc.getRegister(rhs);
    builder.create<as::rtype::Feq>(rd, lhsReg, rhsReg, dataSize);
  } else {
    assert(!lhs.isConstant() ||
           !rhs.isConstant() &&
               "Binary equality must have at least one non-constant operand. "
               "It is guaranteed by `OutlineConstant` pass.");

    if (lhs.isConstant() | rhs.isConstant()) {
      // use seqz
      if (lhs.isConstant())
        std::swap(lhs, rhs);

      auto *rhsImm = getImmediate(rhs.getDefiningInst<ir::inst::Constant>());
      assert(rhsImm && rhsImm->isZero() &&
             "Expected a constant immediate 0 for rhs operand");

      auto lhsReg = regAlloc.getRegister(lhs);
      builder.create<as::pseudo::Seqz>(rd, lhsReg);
    } else {
      // use rtype
      // sub rd, lhs, rhs
      // seqz rd, rd
      auto lhsReg = regAlloc.getRegister(lhs);
      auto rhsReg = regAlloc.getRegister(rhs);

      builder.create<as::rtype::Sub>(rd, lhsReg, rhsReg, dataSize);
      builder.create<as::pseudo::Seqz>(rd, rd);
    }
  }

  return utils::LogicalResult::success();
}

utils::LogicalResult translateNeq(as::AsmBuilder &builder,
                                  FunctionTranslater &translater,
                                  ir::inst::Binary inst) {
  translateEq(builder, translater, inst);
  auto &regAlloc = translater.regAlloc();
  auto dest = inst.getResult();
  auto rd = regAlloc.getRegister(dest);
  // xori rd, rd, 1
  createLogicalNot(builder, rd);
  return utils::LogicalResult::success();
}

utils::LogicalResult translateLt(as::AsmBuilder &builder,
                                 FunctionTranslater &translater,
                                 ir::inst::Binary inst) {
  auto &regAlloc = translater.regAlloc();
  auto lhs = inst.getLhs();
  auto rhs = inst.getRhs();
  auto dest = inst.getResult();
  auto rd = regAlloc.getRegister(dest);
  auto dataSize = getDataSize(lhs.getType());

  if (lhs.getType().isa<ir::FloatT>()) {
    assert(!lhs.isConstant() && !rhs.isConstant() &&
           "Binary float less than operands must not be constant. "
           "It is guaranteed by `OutlineConstant` pass.");
    auto lhsReg = regAlloc.getRegister(lhs);
    auto rhsReg = regAlloc.getRegister(rhs);
    builder.create<as::rtype::Flt>(rd, lhsReg, rhsReg, dataSize);
  } else {
    assert(!lhs.isConstant() && "Binary less than lhs must not be constant. "
                                "It is guaranteed by `OutlineConstant` pass.");

    if (rhs.isConstant()) {
      // use itype
      auto *rhsImm = getImmediate(rhs.getDefiningInst<ir::inst::Constant>());
      assert(rhsImm && "Expected a constant immediate for rhs operand");
      auto lhsReg = regAlloc.getRegister(lhs);

      auto isSigned = rhs.getType().cast<ir::IntT>().isSigned();
      builder.create<as::itype::Slti>(rd, lhsReg, rhsImm, isSigned);
    } else {
      // use rtype
      auto lhsReg = regAlloc.getRegister(lhs);
      auto rhsReg = regAlloc.getRegister(rhs);

      auto isSigned = rhs.getType().cast<ir::IntT>().isSigned();
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

  auto &regAlloc = translater.regAlloc();
  auto lhs = inst.getLhs();
  auto rhs = inst.getRhs();
  auto dest = inst.getResult();
  auto rd = regAlloc.getRegister(dest);
  auto dataSize = getDataSize(lhs.getType());

  if (lhs.getType().isa<ir::FloatT>()) {
    assert(!lhs.isConstant() && !rhs.isConstant() &&
           "Binary float less than or equal operands must not be constant. "
           "It is guaranteed by `OutlineConstant` pass.");
    auto lhsReg = regAlloc.getRegister(lhs);
    auto rhsReg = regAlloc.getRegister(rhs);
    builder.create<as::rtype::Flt>(rd, rhsReg, lhsReg, dataSize);
  } else {
    assert(!rhs.isConstant() &&
           "Binary less than or equal rhs must not be constant. ");

    if (lhs.isConstant()) {
      // use itype
      auto *lhsImm = getImmediate(lhs.getDefiningInst<ir::inst::Constant>());
      assert(lhsImm && "Expected a constant immediate for lhs operand");
      auto rhsReg = regAlloc.getRegister(rhs);

      auto isSigned = lhs.getType().cast<ir::IntT>().isSigned();
      builder.create<as::itype::Slti>(rd, rhsReg, lhsImm, isSigned);
    } else {
      // use rtype
      auto lhsReg = regAlloc.getRegister(lhs);
      auto rhsReg = regAlloc.getRegister(rhs);

      auto isSigned = lhs.getType().cast<ir::IntT>().isSigned();
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

  auto &regAlloc = translater.regAlloc();
  auto lhs = inst.getLhs();
  auto rhs = inst.getRhs();
  auto dest = inst.getResult();
  auto rd = regAlloc.getRegister(dest);
  auto dataSize = getDataSize(lhs.getType());

  if (lhs.getType().isa<ir::FloatT>()) {
    assert(!lhs.isConstant() && !rhs.isConstant() &&
           "Binary float greater than operands must not be constant. "
           "It is guaranteed by `OutlineConstant` pass.");
    auto lhsReg = regAlloc.getRegister(lhs);
    auto rhsReg = regAlloc.getRegister(rhs);
    builder.create<as::rtype::Flt>(rd, rhsReg, lhsReg, dataSize);
  } else {
    assert(!rhs.isConstant() &&
           "Binary greater than rhs must not be constant. ");

    if (lhs.isConstant()) {
      // use itype
      auto *lhsImm = getImmediate(lhs.getDefiningInst<ir::inst::Constant>());
      assert(lhsImm && "Expected a constant immediate for lhs operand");
      auto rhsReg = regAlloc.getRegister(rhs);

      auto isSigned = lhs.getType().cast<ir::IntT>().isSigned();
      builder.create<as::itype::Slti>(rd, rhsReg, lhsImm, isSigned);
    } else {
      // use rtype
      auto lhsReg = regAlloc.getRegister(lhs);
      auto rhsReg = regAlloc.getRegister(rhs);

      auto isSigned = lhs.getType().cast<ir::IntT>().isSigned();
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

  auto &regAlloc = translater.regAlloc();
  auto lhs = inst.getLhs();
  auto rhs = inst.getRhs();
  auto dest = inst.getResult();
  auto rd = regAlloc.getRegister(dest);
  auto dataSize = getDataSize(lhs.getType());

  if (lhs.getType().isa<ir::FloatT>()) {
    assert(!lhs.isConstant() && !rhs.isConstant() &&
           "Binary float less than operands must not be constant. "
           "It is guaranteed by `OutlineConstant` pass.");
    auto lhsReg = regAlloc.getRegister(lhs);
    auto rhsReg = regAlloc.getRegister(rhs);
    builder.create<as::rtype::Flt>(rd, lhsReg, rhsReg, dataSize);
  } else {
    assert(!lhs.isConstant() && "Binary less than lhs must not be constant. "
                                "It is guaranteed by `OutlineConstant` pass.");

    if (rhs.isConstant()) {
      // use itype
      auto *rhsImm = getImmediate(rhs.getDefiningInst<ir::inst::Constant>());
      assert(rhsImm && "Expected a constant immediate for rhs operand");
      auto lhsReg = regAlloc.getRegister(lhs);

      auto isSigned = rhs.getType().cast<ir::IntT>().isSigned();
      builder.create<as::itype::Slti>(rd, lhsReg, rhsImm, isSigned);
    } else {
      // use rtype
      auto lhsReg = regAlloc.getRegister(lhs);
      auto rhsReg = regAlloc.getRegister(rhs);

      auto isSigned = rhs.getType().cast<ir::IntT>().isSigned();
      builder.create<as::rtype::Slt>(rd, lhsReg, rhsReg, isSigned);
    }
  }

  // xori rd, rd, 1
  createLogicalNot(builder, rd);
  return utils::LogicalResult::success();
}

} // namespace

} // namespace kecc
