#include "kecc/asm/Asm.h"
#include "kecc/asm/AsmInstruction.h"
#include "kecc/asm/Register.h"
#include "kecc/ir/IRInstructions.h"
#include "kecc/ir/IRTypes.h"
#include "kecc/translate/IRTranslater.h"
#include "kecc/utils/LogicalResult.h"
#include "llvm/ADT/APFloat.h"

namespace kecc {

namespace {

static void loadInt32(as::AsmBuilder &builder, as::Register rd,
                      std::int32_t value);

static void loadInt64(as::AsmBuilder &builder, FunctionTranslater &translater,
                      as::Register rd, std::int64_t value);

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

as::Immediate *getImmOrLoad(as::AsmBuilder &builder, as::Register rd,
                            std::int32_t value) {
  static constexpr std::int32_t HI12 = (1 << 11) - 1;
  static constexpr std::int32_t LO12 = -(1 << 11);

  if (LO12 <= value && value <= HI12) {
    return getImmediate(value);
  } else {
    loadInt32(builder, rd, value);
    return nullptr; // indicates that a load was created
  }
}

namespace {

void loadInt(as::AsmBuilder &builder, FunctionTranslater &translater,
             as::Register rd, std::int64_t value, ir::IntT intT) {
  if (intT.getBitWidth() <= 32 || (static_cast<std::int32_t>(value) == value)) {
    // If the value can fit in a 32-bit signed integer, treat it as a small int.
    loadInt32(builder, rd, static_cast<std::int32_t>(value));
  } else {
    // Otherwise, treat it as a big int.
    loadInt64(builder, translater, rd, value);
  }
}

void loadInt32(as::AsmBuilder &builder, as::Register rd, std::int32_t value) {
  static constexpr std::int32_t HI12 = (1 << 11) - 1;
  static constexpr std::int32_t LO12 = -(1 << 11);

  if (LO12 <= value && value <= HI12) {
    builder.create<as::itype::Addi>(rd, as::Register::zero(),
                                    getImmediate(value), as::DataSize::word());
  } else {
    std::uint32_t hi12 = (value >> 12);
    std::uint32_t lo12 = value & ((1 << 12) - 1);
    bool isNegative = lo12 >> 11;
    if (isNegative) {
      hi12 += 1;
    }

    builder.create<as::utype::Lui>(rd, getImmediate(hi12));
    if (lo12 != 0) {
      builder.create<as::itype::Addi>(rd, rd, getImmediate(lo12),
                                      as::DataSize::word());
    }
  }
}

void loadInt64(as::AsmBuilder &builder, FunctionTranslater &translater,
               as::Register rd, std::int64_t value) {
  // If the low bits are zero, shift the whole value to the right, treat it as a
  // small int and load it, then shift it back to the left.

  std::int64_t shiftedValue = value;
  size_t shiftRight = 0;
  while (!(shiftedValue & 1)) {
    shiftedValue >>= 1;
    shiftRight++;
  }

  if (static_cast<std::int32_t>(shiftedValue) == shiftedValue) {
    // If the value can fit in a 32-bit signed integer, treat it as a small int.
    loadInt32(builder, rd, static_cast<std::int32_t>(shiftedValue));

    if (shiftRight > 0) {
      builder.create<as::itype::Slli>(rd, rd, getImmediate(shiftRight),
                                      as::DataSize::word());
    }
  } else {
    auto [label, dataSize] = translater.getConstantLabel(value);
    builder.create<as::pseudo::La>(rd, label);
    builder.create<as::itype::Load>(rd, rd, getImmediate(0), dataSize, true);
  }
}

utils::LogicalResult translateFloat(as::AsmBuilder &builder,
                                    FunctionTranslater &translater,
                                    llvm::APFloat value, ir::FloatT floatT,
                                    as::Register rd) {
  // must use temporary integer register

  TranslateContext *context = translater.getTranslateContext();
  auto tempReg = context->getTempRegisters()[0];

  // load float address to temporary register
  auto [label, dataSize] = translater.getConstantLabel(
      ir::ConstantFloatAttr::get(translater.getModule()->getContext(), value));

  builder.create<as::pseudo::La>(tempReg, label);

  // load float value from address
  builder.create<as::itype::Load>(rd, tempReg, getImmediate(0), dataSize, true);
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

} // namespace kecc
