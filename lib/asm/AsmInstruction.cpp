#include "kecc/asm/AsmInstruction.h"
#include "kecc/ir/Interface.h"
#include "llvm/ADT/StringRef.h"
#include <format>

DEFINE_KECC_TYPE_ID(kecc::as::CommentLine)
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

std::string Immediate::toString() const {
  std::string result;
  llvm::raw_string_ostream os(result);
  print(os);
  return result;
}
void ValueImmediate::print(llvm::raw_ostream &os) const { os << getValue(); }
void RelocationImmediate::print(llvm::raw_ostream &os) const {
  std::string func;
  switch (relocation) {
  case RelocationFunction::Hi20:
    func = "%hi";
    break;
  case RelocationFunction::Lo12:
    func = "%lo";
    break;
  }
  os << std::format("{}({})", func, label);
}

void Instruction::remove() { getNode()->remove(); }

CommentLine::CommentLine(llvm::StringRef comment)
    : Base(TypeID::get<CommentLine>()) {
  setComment(comment);
}

bool RType::classof(const Instruction *inst) {
  return llvm::isa<rtype::Add, rtype::Sub, rtype::Sll, rtype::Srl, rtype::Sra,
                   rtype::Mul, rtype::Div, rtype::Rem, rtype::Slt, rtype::Xor,
                   rtype::Or, rtype::And, rtype::Fadd, rtype::Fsub, rtype::Fmul,
                   rtype::Fdiv, rtype::Feq, rtype::Flt, rtype::FmvIntToFloat,
                   rtype::FmvFloatToInt, rtype::FcvtIntToFloat,
                   rtype::FcvtFloatToInt, rtype::FcvtFloatToFloat>(inst);
}

void RType::printInner(llvm::raw_ostream &os) const {
  auto roundingMode = isa<rtype::FcvtFloatToInt>() ? ",rtz" : "";
  os << std::format("{}\t{},{}{}{}", toString(), rd.toString(), rs1.toString(),
                    rs2 ? std::format(",{}", rs2->toString()) : "",
                    roundingMode);
}

namespace rtype {
Add::Add(Register rd, Register rs1, std::optional<Register> rs2,
         DataSize dataSize)
    : Base(TypeID::get<Add>(), rd, rs1, rs2), dataSize(dataSize) {}

std::string Add::toString() const {
  return std::format("add{}", dataSize.isWord() ? "w" : "");
}

Sub::Sub(Register rd, Register rs1, std::optional<Register> rs2,
         DataSize dataSize)
    : Base(TypeID::get<Sub>(), rd, rs1, rs2), dataSize(dataSize) {}

std::string Sub::toString() const {
  return std::format("sub{}", dataSize.isWord() ? "w" : "");
}

Sll::Sll(Register rd, Register rs1, std::optional<Register> rs2,
         DataSize dataSize)
    : Base(TypeID::get<Sll>(), rd, rs1, rs2), dataSize(dataSize) {}

std::string Sll::toString() const {
  return std::format("sll{}", dataSize.isWord() ? "w" : "");
}

Srl::Srl(Register rd, Register rs1, std::optional<Register> rs2,
         DataSize dataSize)
    : Base(TypeID::get<Srl>(), rd, rs1, rs2), dataSize(dataSize) {}

std::string Srl::toString() const {
  return std::format("srl{}", dataSize.isWord() ? "w" : "");
}

Sra::Sra(Register rd, Register rs1, std::optional<Register> rs2,
         DataSize dataSize)
    : Base(TypeID::get<Sra>(), rd, rs1, rs2), dataSize(dataSize) {}

std::string Sra::toString() const {
  return std::format("sra{}", dataSize.isWord() ? "w" : "");
}

Mul::Mul(Register rd, Register rs1, std::optional<Register> rs2,
         DataSize dataSize)
    : Base(TypeID::get<Mul>(), rd, rs1, rs2), dataSize(dataSize) {}

std::string Mul::toString() const {
  return std::format("mul{}", dataSize.isWord() ? "w" : "");
}

Div::Div(Register rd, Register rs1, std::optional<Register> rs2,
         DataSize dataSize, bool isSigned)
    : Base(TypeID::get<Div>(), rd, rs1, rs2), dataSize(dataSize),
      isSignedV(isSigned) {}

std::string Div::toString() const {
  return std::format("div{}{}", isSigned() ? "" : "u",
                     dataSize.isWord() ? "w" : "");
}

Rem::Rem(Register rd, Register rs1, std::optional<Register> rs2,
         DataSize dataSize, bool isSigned)
    : Base(TypeID::get<Rem>(), rd, rs1, rs2), dataSize(dataSize),
      isSignedV(isSigned) {}

std::string Rem::toString() const {
  return std::format("rem{}{}", isSigned() ? "" : "u",
                     dataSize.isWord() ? "w" : "");
}

Slt::Slt(Register rd, Register rs1, std::optional<Register> rs2, bool isSigned)
    : Base(TypeID::get<Slt>(), rd, rs1, rs2), isSignedV(isSigned) {}

std::string Slt::toString() const {
  return std::format("slt{}", isSigned() ? "" : "u");
}

Xor::Xor(Register rd, Register rs1, std::optional<Register> rs2)
    : Base(TypeID::get<Xor>(), rd, rs1, rs2) {}

std::string Xor::toString() const { return "xor"; }

Or::Or(Register rd, Register rs1, std::optional<Register> rs2)
    : Base(TypeID::get<Or>(), rd, rs1, rs2) {}

std::string Or::toString() const { return "or"; }

And::And(Register rd, Register rs1, std::optional<Register> rs2)
    : Base(TypeID::get<And>(), rd, rs1, rs2) {}

std::string And::toString() const { return "and"; }

Fadd::Fadd(Register rd, Register rs1, std::optional<Register> rs2,
           DataSize dataSize)
    : Base(TypeID::get<Fadd>(), rd, rs1, rs2), dataSize(dataSize) {}

std::string Fadd::toString() const {
  return std::format("fadd.{}", dataSize.toString());
}

Fsub::Fsub(Register rd, Register rs1, std::optional<Register> rs2,
           DataSize dataSize)
    : Base(TypeID::get<Fsub>(), rd, rs1, rs2), dataSize(dataSize) {}

std::string Fsub::toString() const {
  return std::format("fsub.{}", dataSize.toString());
}

Fmul::Fmul(Register rd, Register rs1, std::optional<Register> rs2,
           DataSize dataSize)
    : Base(TypeID::get<Fmul>(), rd, rs1, rs2), dataSize(dataSize) {}

std::string Fmul::toString() const {
  return std::format("fmul.{}", dataSize.toString());
}

Fdiv::Fdiv(Register rd, Register rs1, std::optional<Register> rs2,
           DataSize dataSize)
    : Base(TypeID::get<Fdiv>(), rd, rs1, rs2), dataSize(dataSize) {}

std::string Fdiv::toString() const {
  return std::format("fdiv.{}", dataSize.toString());
}

Feq::Feq(Register rd, Register rs1, std::optional<Register> rs2,
         DataSize dataSize)
    : Base(TypeID::get<Feq>(), rd, rs1, rs2), dataSize(dataSize) {}

std::string Feq::toString() const {
  return std::format("feq.{}", dataSize.toString());
}

Flt::Flt(Register rd, Register rs1, std::optional<Register> rs2,
         DataSize dataSize)
    : Base(TypeID::get<Flt>(), rd, rs1, rs2), dataSize(dataSize) {}

std::string Flt::toString() const {
  return std::format("flt.{}", dataSize.toString());
}

Fle::Fle(Register rd, Register rs1, std::optional<Register> rs2,
         DataSize dataSize)
    : Base(TypeID::get<Fle>(), rd, rs1, rs2), dataSize(dataSize) {}

std::string Fle::toString() const {
  return std::format("fle.{}", dataSize.toString());
}

FmvIntToFloat::FmvIntToFloat(Register rd, Register rs1,
                             std::optional<Register> rs2,
                             DataSize floatDataSize)
    : Base(TypeID::get<FmvIntToFloat>(), rd, rs1, rs2),
      floatDataSize(floatDataSize) {}

std::string FmvIntToFloat::toString() const {
  assert(floatDataSize.isFloat() &&
         "FmvIntToFloat should only be used with float data size");
  return std::format("fmv.{}.x", floatDataSize.isSinglePrecision() ? "w" : "d");
}

FmvFloatToInt::FmvFloatToInt(Register rd, Register rs1,
                             std::optional<Register> rs2,
                             DataSize floatDataSize)
    : Base(TypeID::get<FmvFloatToInt>(), rd, rs1, rs2),
      floatDataSize(floatDataSize) {}

std::string FmvFloatToInt::toString() const {
  assert(floatDataSize.isFloat() &&
         "FmvFloatToInt should only be used with float data size");
  return std::format("fmv.x.{}", floatDataSize.isSinglePrecision() ? "w" : "d");
}

FcvtIntToFloat::FcvtIntToFloat(Register rd, Register rs1,
                               std::optional<Register> rs2,
                               DataSize intDataSize, DataSize floatDataSize,
                               bool isSigned)
    : Base(TypeID::get<FcvtIntToFloat>(), rd, rs1, rs2),
      intDataSize(intDataSize), floatDataSize(floatDataSize),
      isSignedV(isSigned) {}

std::string FcvtIntToFloat::toString() const {
  assert(floatDataSize.isFloat() && intDataSize.isInt() &&
         "FcvtFloatToInt should only be used with float and int data sizes");
  return std::format("fcvt.{}.{}{}", floatDataSize.toString(),
                     intDataSize.isWord() ? "w" : "l", isSigned() ? "" : "u");
}

FcvtFloatToInt::FcvtFloatToInt(Register rd, Register rs1,
                               std::optional<Register> rs2,
                               DataSize floatDataSize, DataSize intDataSize,
                               bool isSigned)
    : Base(TypeID::get<FcvtFloatToInt>(), rd, rs1, rs2),
      floatDataSize(floatDataSize), intDataSize(intDataSize),
      isSignedV(isSigned) {}

std::string FcvtFloatToInt::toString() const {
  assert(intDataSize.isInt() && floatDataSize.isFloat() &&
         "FcvtIntToFloat should only be used with int and float data sizes");
  return std::format("fcvt.{}{}.{}", intDataSize.isWord() ? "w" : "l",
                     isSigned() ? "" : "u", floatDataSize.toString());
}

FcvtFloatToFloat::FcvtFloatToFloat(Register rd, Register rs1,
                                   std::optional<Register> rs2,
                                   DataSize fromDataSize, DataSize toDataSize)
    : Base(TypeID::get<FcvtFloatToFloat>(), rd, rs1, rs2), from(fromDataSize),
      to(toDataSize) {}

std::string FcvtFloatToFloat::toString() const {
  assert(from.isFloat() && to.isFloat() &&
         "FcvtFloatToFloat should only be used with float data sizes");
  return std::format("fcvt.{}.{}", to.toString(), from.toString());
}
} // namespace rtype

IType::~IType() { delete imm; }
bool IType::classof(const Instruction *inst) {
  return llvm::isa<itype::Load, itype::Addi, itype::Xori, itype::Ori,
                   itype::Andi, itype::Slli, itype::Srli, itype::Srai,
                   itype::Slti>(inst);
}
void IType::printInner(llvm::raw_ostream &os) const {
  if (isa<itype::Load>())
    os << std::format("{}\t{},{}({})", toString(), rd.toString(),
                      imm->toString(), rs1.toString());
  else
    os << std::format("{}\t{},{},{}", toString(), rd.toString(), rs1.toString(),
                      imm->toString());
}

namespace itype {

Load::Load(Register rd, Register rs1, Immediate *imm, DataSize dataSize,
           bool isSigned)
    : Base(TypeID::get<Load>(), rd, rs1, imm), dataSize(dataSize),
      isSignedV(isSigned) {}

std::string Load::toString() const {
  if (dataSize.isInt())
    return std::format("l{}{}", dataSize.toString(),
                       (isSigned() || dataSize.isDouble()) ? "" : "u");
  return std::format("fl{}", dataSize.isSinglePrecision() ? "w" : "d");
}

Addi::Addi(Register rd, Register rs1, Immediate *imm, DataSize dataSize)
    : Base(TypeID::get<Addi>(), rd, rs1, imm), dataSize(dataSize) {}

std::string Addi::toString() const {
  return std::format("addi{}", dataSize.isWord() ? "w" : "");
}

Xori::Xori(Register rd, Register rs1, Immediate *imm)
    : Base(TypeID::get<Xori>(), rd, rs1, imm) {}
std::string Xori::toString() const { return "xori"; }

Ori::Ori(Register rd, Register rs1, Immediate *imm)
    : Base(TypeID::get<Ori>(), rd, rs1, imm) {}
std::string Ori::toString() const { return "ori"; }

Andi::Andi(Register rd, Register rs1, Immediate *imm)
    : Base(TypeID::get<Andi>(), rd, rs1, imm) {}
std::string Andi::toString() const { return "andi"; }

Slli::Slli(Register rd, Register rs1, Immediate *imm, DataSize dataSize)
    : Base(TypeID::get<Slli>(), rd, rs1, imm), dataSize(dataSize) {}

std::string Slli::toString() const {
  return std::format("slli{}", dataSize.isWord() ? "w" : "");
}

Srli::Srli(Register rd, Register rs1, Immediate *imm, DataSize dataSize)
    : Base(TypeID::get<Srli>(), rd, rs1, imm), dataSize(dataSize) {}

std::string Srli::toString() const {
  return std::format("srli{}", dataSize.isWord() ? "w" : "");
}

Srai::Srai(Register rd, Register rs1, Immediate *imm, DataSize dataSize)
    : Base(TypeID::get<Srai>(), rd, rs1, imm), dataSize(dataSize) {}

std::string Srai::toString() const {
  return std::format("srai{}", dataSize.isWord() ? "w" : "");
}

Slti::Slti(Register rd, Register rs1, Immediate *imm, bool isSigned)
    : Base(TypeID::get<Slti>(), rd, rs1, imm), isSignedV(isSigned) {}

std::string Slti::toString() const {
  return std::format("slti{}", isSigned() ? "" : "u");
}
} // namespace itype

SType::~SType() { delete imm; }
bool SType::classof(const Instruction *inst) {
  return llvm::isa<stype::Store>(inst);
}
void SType::printInner(llvm::raw_ostream &os) const {
  os << std::format("{}\t{},{}({})", toString(), rs2.toString(),
                    imm->toString(), rs1.toString());
}

namespace stype {

Store::Store(Register base, Register src, Immediate *imm, DataSize dataSize)
    : Base(TypeID::get<Store>(), base, src, imm), dataSize(dataSize) {}

std::string Store::toString() const {
  if (dataSize.isInt())
    return std::format("s{}", dataSize.toString());
  return std::format("fs{}", dataSize.isSinglePrecision() ? "w" : "d");
}

} // namespace stype

bool BType::classof(const Instruction *inst) {
  return llvm::isa<btype::Beq, btype::Bne, btype::Blt, btype::Bge>(inst);
}
void BType::printInner(llvm::raw_ostream &os) const {
  os << std::format("{}\t{},{}, {}", toString(), rs1.toString(), rs2.toString(),
                    imm);
}

namespace btype {

Beq::Beq(Register rs1, Register rs2, llvm::StringRef imm)
    : Base(TypeID::get<Beq>(), rs1, rs2, imm) {}
std::string Beq::toString() const { return "beq"; }

Bne::Bne(Register rs1, Register rs2, llvm::StringRef imm)
    : Base(TypeID::get<Bne>(), rs1, rs2, imm) {}
std::string Bne::toString() const { return "bne"; }

Blt::Blt(Register rs1, Register rs2, llvm::StringRef imm, bool isSigned)
    : Base(TypeID::get<Blt>(), rs1, rs2, imm), isSignedV(isSigned) {}
std::string Blt::toString() const {
  return std::format("blt{}", isSigned() ? "" : "u");
}

Bge::Bge(Register rs1, Register rs2, llvm::StringRef imm, bool isSigned)
    : Base(TypeID::get<Beq>(), rs1, rs2, imm), isSignedV(isSigned) {}
std::string Bge::toString() const {
  return std::format("bge{}", isSigned() ? "" : "u");
}

} // namespace btype

UType::~UType() { delete imm; }
bool UType::classof(const Instruction *inst) {
  return llvm::isa<utype::Lui>(inst);
}
void UType::printInner(llvm::raw_ostream &os) const {
  os << std::format("{}\t{},{}", toString(), rd.toString(), imm->toString());
}

namespace utype {
Lui::Lui(Register rd, Immediate *imm) : Base(TypeID::get<Lui>(), rd, imm) {}
std::string Lui::toString() const { return "lui"; }
} // namespace utype

bool Pseudo::classof(const Instruction *inst) {
  return llvm::isa<pseudo::La, pseudo::Li, pseudo::Mv, pseudo::Fmv, pseudo::Neg,
                   pseudo::SextW, pseudo::Seqz, pseudo::Snez, pseudo::Fneg,
                   pseudo::J, pseudo::Jr, pseudo::Jalr, pseudo::Ret,
                   pseudo::Call>(inst);
}

void Pseudo::printInner(llvm::raw_ostream &os) const { os << toString(); }

namespace pseudo {

La::La(Register rd, llvm::StringRef symbol)
    : Base(TypeID::get<La>()), rd(rd), symbol(symbol) {}

std::string La::toString() const {
  return std::format("la\t{},{}", rd.toString(), symbol);
}

Li::Li(Register rd, std::size_t imm)
    : Base(TypeID::get<Li>()), rd(rd), imm(imm) {}

std::string Li::toString() const {
  return std::format("li\t{},{}", rd.toString(), imm);
}

Mv::Mv(Register rd, Register rs) : Base(TypeID::get<Mv>()), rd(rd), rs(rs) {}

std::string Mv::toString() const {
  return std::format("mv\t{},{}", rd.toString(), rs.toString());
}

Fmv::Fmv(DataSize dataSize, Register rd, Register rs)
    : Base(TypeID::get<Fmv>()), dataSize(dataSize), rd(rd), rs(rs) {}

std::string Fmv::toString() const {
  return std::format("fmv.{}\t{},{}", dataSize.toString(), rd.toString(),
                     rs.toString());
}

Not::Not(Register rd, Register rs) : Base(TypeID::get<Not>()), rd(rd), rs(rs) {}

std::string Not::toString() const {
  return std::format("not\t{},{}", rd.toString(), rs.toString());
}

Neg::Neg(DataSize dataSize, Register rd, Register rs)
    : Base(TypeID::get<Neg>()), dataSize(dataSize), rd(rd), rs(rs) {}

std::string Neg::toString() const {
  return std::format("neg{}\t{},{}", dataSize.isWord() ? "w" : "",
                     rd.toString(), rs.toString());
}

SextW::SextW(Register rd, Register rs)
    : Base(TypeID::get<SextW>()), rd(rd), rs(rs) {}

std::string SextW::toString() const {
  return std::format("sext.w\t{},{}", rd.toString(), rs.toString());
}

Seqz::Seqz(Register rd, Register rs)
    : Base(TypeID::get<Seqz>()), rd(rd), rs(rs) {}

std::string Seqz::toString() const {
  return std::format("seqz\t{},{}", rd.toString(), rs.toString());
}

Snez::Snez(Register rd, Register rs)
    : Base(TypeID::get<Snez>()), rd(rd), rs(rs) {}

std::string Snez::toString() const {
  return std::format("snez\t{},{}", rd.toString(), rs.toString());
}

Fneg::Fneg(DataSize dataSize, Register rd, Register rs)
    : Base(TypeID::get<Fneg>()), dataSize(dataSize), rd(rd), rs(rs) {}

std::string Fneg::toString() const {
  return std::format("fneg.{}\t{},{}", dataSize.toString(), rd.toString(),
                     rs.toString());
}

J::J(llvm::StringRef label) : Base(TypeID::get<J>()), label(label) {}

std::string J::toString() const { return std::format("j\t{}", label); }

Jr::Jr(Register rs) : Base(TypeID::get<Jr>()), rs(rs) {}
std::string Jr::toString() const {
  return std::format("jr\t{}", rs.toString());
}

Jalr::Jalr(Register rs) : Base(TypeID::get<Jalr>()), rs(rs) {}
std::string Jalr::toString() const {
  return std::format("jalr\t{}", rs.toString());
}

Ret::Ret() : Base(TypeID::get<Ret>()) {}

std::string Ret::toString() const { return "ret"; }

Call::Call(llvm::StringRef label) : Base(TypeID::get<Call>()), label(label) {}

std::string Call::toString() const { return std::format("call\t{}", label); }

} // namespace pseudo

} // namespace kecc::as
