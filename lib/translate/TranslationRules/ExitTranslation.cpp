#include "kecc/asm/AsmInstruction.h"
#include "kecc/ir/IRTypes.h"
#include "kecc/ir/JumpArg.h"
#include "kecc/translate/IRTranslater.h"

namespace kecc {

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

    static llvm::ArrayRef<as::Register> intReturnRegisters = {
        as::Register::a0(),
        as::Register::a1(),
    };

    static llvm::ArrayRef<as::Register> floatReturnRegisters = {
        as::Register::fa0(),
        as::Register::fa1(),
    };

    llvm::SmallVector<std::pair<as::Register, as::DataSize>> intValueRegisters;
    llvm::SmallVector<std::pair<as::Register, as::DataSize>>
        floatValueRegisters;
    intValueRegisters.reserve(intReturnRegisters.size());
    floatValueRegisters.reserve(floatReturnRegisters.size());

    for (auto idx = 0u; idx < inst.getValueSize(); ++idx) {
      const auto *value = &inst.getValueAsOperand(idx);
      as::Register valueReg = translater.getOperandRegister(value);
      if (value->getType().isa<ir::FloatT>())
        floatValueRegisters.emplace_back(valueReg);
      else
        intValueRegisters.emplace_back(valueReg);
    }

    assert(intValueRegisters.size() <= intReturnRegisters.size() &&
           "Too many integer return values for available registers");
    assert(floatValueRegisters.size() <= floatReturnRegisters.size() &&
           "Too many floating point return values for available registers");

    if (!intValueRegisters.empty()) {
      translater.moveRegisters(
          builder, intValueRegisters,
          intReturnRegisters.take_front(intValueRegisters.size()));
    }

    if (!floatValueRegisters.empty()) {
      translater.moveRegisters(
          builder, floatValueRegisters,
          floatReturnRegisters.take_front(floatValueRegisters.size()));
    }

    translater.writeFunctionEnd(builder);

    // Create a pseudo return instruction
    builder.create<as::pseudo::Ret>();

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

} // namespace kecc
