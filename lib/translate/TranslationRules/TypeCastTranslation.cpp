#include "kecc/asm/AsmInstruction.h"
#include "kecc/ir/IRInstructions.h"
#include "kecc/ir/IRTypes.h"
#include "kecc/translate/IRTranslater.h"

namespace kecc {

static void castIntToInt(as::AsmBuilder &builder, as::Register rd,
                         as::Register srcReg, ir::IntT fromT, ir::IntT toT) {
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

  if (fromBitWidth == toBitWidth) {
    // mv rd, srcReg
    builder.create<as::pseudo::Mv>(rd, srcReg);
  } else if (fromBitWidth < toBitWidth) {
    if (!fromSigned && toSigned) {
      // fill 0 in higher bits
      auto bitDiff = toBitWidth - fromBitWidth;
      builder.create<as::itype::Slli>(rd, srcReg, getImmediate(bitDiff),
                                      as::DataSize::doubleWord());
      builder.create<as::itype::Srli>(rd, rd, getImmediate(bitDiff),
                                      as::DataSize::doubleWord());
    } else {
      // mv rd, srcReg
      builder.create<as::pseudo::Mv>(rd, srcReg);
    }
  } else {
    auto bitDiff = fromBitWidth - toBitWidth;
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

    if (srcType.isa<ir::PointerT>()) {
      if (dstType.isa<ir::IntT>()) {
        castIntToInt(builder, rd, srcReg,
                     ir::IntT::get(
                         srcType.getContext(),
                         srcType.getContext()->getArchitectureBitSize(), false),
                     dstType.cast<ir::IntT>());
      } else if (dstType.isa<ir::PointerT>()) {
        // Pointer to Pointer cast, just move the register
        builder.create<as::pseudo::Mv>(rd, srcReg);
      }

      llvm_unreachable("Unsupported pointer cast type");
    } else if (auto srcIntT = srcType.dyn_cast<ir::IntT>()) {
      if (dstType.isa<ir::PointerT>()) {
        castIntToInt(
            builder, rd, srcReg, srcIntT,
            ir::IntT::get(srcType.getContext(),
                          srcType.getContext()->getArchitectureBitSize(),
                          false));
      } else if (auto dstIntT = dstType.dyn_cast<ir::IntT>()) {
        castIntToInt(builder, rd, srcReg, srcIntT, dstIntT);
      } else if (auto dstFloatT = dstType.dyn_cast<ir::FloatT>()) {
        // FcvIntToFloat
        builder.create<as::rtype::FcvtIntToFloat>(rd, srcReg, std::nullopt,
                                                  fromDataSize, toDataSize,
                                                  srcIntT.isSigned());
      } else {
        llvm_unreachable("Unsupported int cast type");
      }
    } else if (auto srcFloatT = srcType.dyn_cast<ir::FloatT>()) {
      if (auto dstIntT = dstType.dyn_cast<ir::IntT>()) {
        // FcvFloatToInt
        builder.create<as::rtype::FcvtFloatToInt>(rd, srcReg, std::nullopt,
                                                  fromDataSize, toDataSize,
                                                  dstIntT.isSigned());
      } else if (dstType.isa<ir::FloatT>()) {
        // FcvFloatToFloat
        builder.create<as::rtype::FcvtFloatToFloat>(rd, srcReg, std::nullopt,
                                                    fromDataSize, toDataSize);
      } else {
        llvm_unreachable("Unsupported float cast type");
      }
    } else {
      llvm_unreachable("Unsupported type cast");
    }

    return utils::LogicalResult::success();
  }
};

} // namespace kecc
