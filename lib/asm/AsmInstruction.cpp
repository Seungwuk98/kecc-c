#include "kecc/asm/AsmInstruction.h"

DEFINE_KECC_TYPE_ID(kecc::as::rtype::Add)
DEFINE_KECC_TYPE_ID(kecc::as::rtype::Sub)
DEFINE_KECC_TYPE_ID(kecc::as::rtype::Sll)
DEFINE_KECC_TYPE_ID(kecc::as::rtype::Srl)
DEFINE_KECC_TYPE_ID(kecc::as::rtype::Sra)
DEFINE_KECC_TYPE_ID(kecc::as::rtype::Mul)
DEFINE_KECC_TYPE_ID(kecc::as::rtype::Div)
DEFINE_KECC_TYPE_ID(kecc::as::rtype::Rem)
DEFINE_KECC_TYPE_ID(kecc::as::rtype::Slt)
DEFINE_KECC_TYPE_ID(kecc::as::rtype::Xor)
DEFINE_KECC_TYPE_ID(kecc::as::rtype::Or)
DEFINE_KECC_TYPE_ID(kecc::as::rtype::And)
DEFINE_KECC_TYPE_ID(kecc::as::rtype::Fadd)
DEFINE_KECC_TYPE_ID(kecc::as::rtype::Fsub)
DEFINE_KECC_TYPE_ID(kecc::as::rtype::Fmul)
DEFINE_KECC_TYPE_ID(kecc::as::rtype::Fdiv)
DEFINE_KECC_TYPE_ID(kecc::as::rtype::Feq)
DEFINE_KECC_TYPE_ID(kecc::as::rtype::Flt)
DEFINE_KECC_TYPE_ID(kecc::as::rtype::FmvIntToFloat)
DEFINE_KECC_TYPE_ID(kecc::as::rtype::FmvFloatToInt)
DEFINE_KECC_TYPE_ID(kecc::as::rtype::FcvtIntToFloat)
DEFINE_KECC_TYPE_ID(kecc::as::rtype::FcvtFloatToInt)
DEFINE_KECC_TYPE_ID(kecc::as::rtype::FcvtFloatToFloat)

DEFINE_KECC_TYPE_ID(kecc::as::itype::Load)
DEFINE_KECC_TYPE_ID(kecc::as::itype::Addi)
DEFINE_KECC_TYPE_ID(kecc::as::itype::Xori)
DEFINE_KECC_TYPE_ID(kecc::as::itype::Ori)
DEFINE_KECC_TYPE_ID(kecc::as::itype::Andi)
DEFINE_KECC_TYPE_ID(kecc::as::itype::Slli)
DEFINE_KECC_TYPE_ID(kecc::as::itype::Srli)
DEFINE_KECC_TYPE_ID(kecc::as::itype::Srai)
DEFINE_KECC_TYPE_ID(kecc::as::itype::Slti)

DEFINE_KECC_TYPE_ID(kecc::as::stype::Store)

DEFINE_KECC_TYPE_ID(kecc::as::btype::Beq)
DEFINE_KECC_TYPE_ID(kecc::as::btype::Bne)
DEFINE_KECC_TYPE_ID(kecc::as::btype::Blt)
DEFINE_KECC_TYPE_ID(kecc::as::btype::Bge)

DEFINE_KECC_TYPE_ID(kecc::as::utype::Lui)

DEFINE_KECC_TYPE_ID(kecc::as::pseudo::La)
DEFINE_KECC_TYPE_ID(kecc::as::pseudo::Li)
DEFINE_KECC_TYPE_ID(kecc::as::pseudo::Mv)
DEFINE_KECC_TYPE_ID(kecc::as::pseudo::Fmv)
DEFINE_KECC_TYPE_ID(kecc::as::pseudo::Neg)
DEFINE_KECC_TYPE_ID(kecc::as::pseudo::SextW)
DEFINE_KECC_TYPE_ID(kecc::as::pseudo::Seqz)
DEFINE_KECC_TYPE_ID(kecc::as::pseudo::Snez)
DEFINE_KECC_TYPE_ID(kecc::as::pseudo::Fneg)
DEFINE_KECC_TYPE_ID(kecc::as::pseudo::J)
DEFINE_KECC_TYPE_ID(kecc::as::pseudo::Jr)
DEFINE_KECC_TYPE_ID(kecc::as::pseudo::Jalr)
DEFINE_KECC_TYPE_ID(kecc::as::pseudo::Ret)
DEFINE_KECC_TYPE_ID(kecc::as::pseudo::Call)

namespace kecc::as {

bool RType::classof(const Instruction *inst) {
  return llvm::isa<rtype::Add, rtype::Sub, rtype::Sll, rtype::Srl, rtype::Sra,
                   rtype::Mul, rtype::Div, rtype::Rem, rtype::Slt, rtype::Xor,
                   rtype::Or, rtype::And, rtype::Fadd, rtype::Fsub, rtype::Fmul,
                   rtype::Fdiv, rtype::Feq, rtype::Flt, rtype::FmvIntToFloat,
                   rtype::FmvFloatToInt, rtype::FcvtIntToFloat,
                   rtype::FcvtFloatToInt, rtype::FcvtFloatToFloat>(inst);
}

namespace rtype {
Add::Add(Register rd, Register rs1, std::optional<Register> rs2,
         DataSize dataSize)
    : Base(TypeID::get<Add>(), rd, rs1, rs2), dataSize(dataSize) {}

Sub::Sub(Register rd, Register rs1, std::optional<Register> rs2,
         DataSize dataSize)
    : Base(TypeID::get<Sub>(), rd, rs1, rs2), dataSize(dataSize) {}

Sll::Sll(Register rd, Register rs1, std::optional<Register> rs2,
         DataSize dataSize)
    : Base(TypeID::get<Sll>(), rd, rs1, rs2), dataSize(dataSize) {}

Srl::Srl(Register rd, Register rs1, std::optional<Register> rs2,
         DataSize dataSize)
    : Base(TypeID::get<Srl>(), rd, rs1, rs2), dataSize(dataSize) {}

Sra::Sra(Register rd, Register rs1, std::optional<Register> rs2,
         DataSize dataSize)
    : Base(TypeID::get<Sra>(), rd, rs1, rs2), dataSize(dataSize) {}

Mul::Mul(Register rd, Register rs1, std::optional<Register> rs2,
         DataSize dataSize)
    : Base(TypeID::get<Mul>(), rd, rs1, rs2), dataSize(dataSize) {}

Div::Div(Register rd, Register rs1, std::optional<Register> rs2,
         DataSize dataSize, bool isSigned)
    : Base(TypeID::get<Div>(), rd, rs1, rs2), dataSize(dataSize),
      isSignedV(isSigned) {}

Rem::Rem(Register rd, Register rs1, std::optional<Register> rs2,
         DataSize dataSize, bool isSigned)
    : Base(TypeID::get<Rem>(), rd, rs1, rs2), dataSize(dataSize),
      isSignedV(isSigned) {}

Slt::Slt(Register rd, Register rs1, std::optional<Register> rs2, bool isSigned)
    : Base(TypeID::get<Slt>(), rd, rs1, rs2), isSignedV(isSigned) {}

Xor::Xor(Register rd, Register rs1, std::optional<Register> rs2)
    : Base(TypeID::get<Xor>(), rd, rs1, rs2) {}

Or::Or(Register rd, Register rs1, std::optional<Register> rs2)
    : Base(TypeID::get<Or>(), rd, rs1, rs2) {}

And::And(Register rd, Register rs1, std::optional<Register> rs2)
    : Base(TypeID::get<And>(), rd, rs1, rs2) {}

Fadd::Fadd(Register rd, Register rs1, std::optional<Register> rs2,
           DataSize dataSize)
    : Base(TypeID::get<Fadd>(), rd, rs1, rs2), dataSize(dataSize) {}

Fsub::Fsub(Register rd, Register rs1, std::optional<Register> rs2,
           DataSize dataSize)
    : Base(TypeID::get<Fsub>(), rd, rs1, rs2), dataSize(dataSize) {}

Fmul::Fmul(Register rd, Register rs1, std::optional<Register> rs2,
           DataSize dataSize)
    : Base(TypeID::get<Fmul>(), rd, rs1, rs2), dataSize(dataSize) {}

Fdiv::Fdiv(Register rd, Register rs1, std::optional<Register> rs2,
           DataSize dataSize)
    : Base(TypeID::get<Fdiv>(), rd, rs1, rs2), dataSize(dataSize) {}

Feq::Feq(Register rd, Register rs1, std::optional<Register> rs2,
         DataSize dataSize)
    : Base(TypeID::get<Feq>(), rd, rs1, rs2), dataSize(dataSize) {}

Flt::Flt(Register rd, Register rs1, std::optional<Register> rs2,
         DataSize dataSize)
    : Base(TypeID::get<Flt>(), rd, rs1, rs2), dataSize(dataSize) {}

FmvIntToFloat::FmvIntToFloat(Register rd, Register rs1,
                             std::optional<Register> rs2,
                             DataSize floatDataSize)
    : Base(TypeID::get<FmvIntToFloat>(), rd, rs1, rs2),
      floatDataSize(floatDataSize) {}

FmvFloatToInt::FmvFloatToInt(Register rd, Register rs1,
                             std::optional<Register> rs2,
                             DataSize floatDataSize)
    : Base(TypeID::get<FmvFloatToInt>(), rd, rs1, rs2),
      floatDataSize(floatDataSize) {}

FcvtIntToFloat::FcvtIntToFloat(Register rd, Register rs1,
                               std::optional<Register> rs2,
                               DataSize intDataSize, DataSize floatDataSize,
                               bool isSigned)
    : Base(TypeID::get<FcvtIntToFloat>(), rd, rs1, rs2),
      intDataSize(intDataSize), floatDataSize(floatDataSize),
      isSignedV(isSigned) {}

FcvtFloatToInt::FcvtFloatToInt(Register rd, Register rs1,
                               std::optional<Register> rs2,
                               DataSize floatDataSize, DataSize intDataSize,
                               bool isSigned)
    : Base(TypeID::get<FcvtFloatToInt>(), rd, rs1, rs2),
      floatDataSize(floatDataSize), intDataSize(intDataSize),
      isSignedV(isSigned) {}

FcvtFloatToFloat::FcvtFloatToFloat(Register rd, Register rs1,
                                   std::optional<Register> rs2,
                                   DataSize fromDataSize, DataSize toDataSize)
    : Base(TypeID::get<FcvtFloatToFloat>(), rd, rs1, rs2), from(fromDataSize),
      to(toDataSize) {}
} // namespace rtype

IType::~IType() { delete imm; }
bool IType::classof(const Instruction *inst) {
  return llvm::isa<itype::Load, itype::Addi, itype::Xori, itype::Ori,
                   itype::Andi, itype::Slli, itype::Srli, itype::Srai,
                   itype::Slti>(inst);
}

namespace itype {

Load::Load(Register rd, Register rs1, Immediate *imm, DataSize dataSize,
           bool isSigned)
    : Base(TypeID::get<Load>(), rd, rs1, imm), dataSize(dataSize),
      isSignedV(isSigned) {}

Addi::Addi(Register rd, Register rs1, Immediate *imm, DataSize dataSize)
    : Base(TypeID::get<Addi>(), rd, rs1, imm), dataSize(dataSize) {}

Slli::Slli(Register rd, Register rs1, Immediate *imm, DataSize dataSize)
    : Base(TypeID::get<Slli>(), rd, rs1, imm), dataSize(dataSize) {}

Srli::Srli(Register rd, Register rs1, Immediate *imm, DataSize dataSize)
    : Base(TypeID::get<Srli>(), rd, rs1, imm), dataSize(dataSize) {}

Srai::Srai(Register rd, Register rs1, Immediate *imm, DataSize dataSize)
    : Base(TypeID::get<Srai>(), rd, rs1, imm), dataSize(dataSize) {}

Slti::Slti(Register rd, Register rs1, Immediate *imm, bool isSigned)
    : Base(TypeID::get<Slti>(), rd, rs1, imm), isSignedV(isSigned) {}

} // namespace itype

SType::~SType() { delete imm; }
bool SType::classof(const Instruction *inst) {
  return llvm::isa<stype::Store>(inst);
}

namespace stype {

Store::Store(Register rs1, Register rs2, Immediate *imm, DataSize dataSize)
    : Base(TypeID::get<Store>(), rs1, rs2, imm), dataSize(dataSize) {}

} // namespace stype

BType::~BType() { delete imm; }
bool BType::classof(const Instruction *inst) {
  return llvm::isa<btype::Beq, btype::Bne, btype::Blt, btype::Bge>(inst);
}

namespace btype {

Blt::Blt(Register rs1, Register rs2, Immediate *imm, bool isSigned)
    : Base(TypeID::get<Blt>(), rs1, rs2, imm), isSignedV(isSigned) {}

Bge::Bge(Register rs1, Register rs2, Immediate *imm, bool isSigned)
    : Base(TypeID::get<Beq>(), rs1, rs2, imm), isSignedV(isSigned) {}

} // namespace btype

UType::~UType() { delete imm; }
bool UType::classof(const Instruction *inst) {
  return llvm::isa<utype::Lui>(inst);
}

bool Pseudo::classof(const Instruction *inst) {
  return llvm::isa<pseudo::La, pseudo::Li, pseudo::Mv, pseudo::Fmv, pseudo::Neg,
                   pseudo::SextW, pseudo::Seqz, pseudo::Snez, pseudo::Fneg,
                   pseudo::J, pseudo::Jr, pseudo::Jalr, pseudo::Ret,
                   pseudo::Call>(inst);
}

namespace pseudo {

La::La(Register rd, Immediate *imm)
    : Base(TypeID::get<La>()), rd(rd), imm(imm) {}

Li::Li(Register rd, std::size_t imm)
    : Base(TypeID::get<Li>()), rd(rd), imm(imm) {}

Mv::Mv(Register rd, Register rs) : Base(TypeID::get<Mv>()), rd(rd), rs(rs) {}

Fmv::Fmv(DataSize dataSize, Register rd, Register rs)
    : Base(TypeID::get<Fmv>()), dataSize(dataSize), rd(rd), rs(rs) {}

Neg::Neg(DataSize dataSize, Register rd, Register rs)
    : Base(TypeID::get<Neg>()), dataSize(dataSize), rd(rd), rs(rs) {}

SextW::SextW(Register rd, Register rs)
    : Base(TypeID::get<SextW>()), rd(rd), rs(rs) {}

Seqz::Seqz(Register rd, Register rs)
    : Base(TypeID::get<Seqz>()), rd(rd), rs(rs) {}

Snez::Snez(Register rd, Register rs)
    : Base(TypeID::get<Snez>()), rd(rd), rs(rs) {}

Fneg::Fneg(DataSize dataSize, Register rd, Register rs)
    : Base(TypeID::get<Fneg>()), dataSize(dataSize), rd(rd), rs(rs) {}

J::J(llvm::StringRef label) : Base(TypeID::get<J>()), label(label) {}

Jr::Jr(Register rs) : Base(TypeID::get<Jr>()), rs(rs) {}

Jalr::Jalr(Register rs) : Base(TypeID::get<Jalr>()), rs(rs) {}

Call::Call(llvm::StringRef label) : Base(TypeID::get<Call>()), label(label) {}

} // namespace pseudo

} // namespace kecc::as
