#include "kecc/ir/IRInstructions.h"
#include "kecc/translate/IRTranslater.h"

namespace kecc {

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

} // namespace kecc
