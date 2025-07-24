#ifndef KECC_ASM_INSTRUCTION_H
#define KECC_ASM_INSTRUCTION_H

#include "kecc/asm/Asm.h"
#include "kecc/asm/Register.h"
#include "kecc/utils/MLIR.h"
#include "llvm/Support/TrailingObjects.h"

namespace kecc::as {

class Immediate {
public:
  virtual ~Immediate() = default;
  enum class Kind {
    Value,
    Relocation,
  };

  Kind getKind() const { return kind; }

  virtual void print(llvm::raw_ostream &os) const = 0;

protected:
  Immediate(Kind kind) : kind(kind) {}

private:
  Kind kind;
};

class ValueImmediate : public Immediate {
public:
  ValueImmediate(std::size_t value) : Immediate(Kind::Value), value(value) {}

  std::size_t getValue() const { return value; }

  static bool classof(const Immediate *imm) {
    return imm->getKind() == Kind::Value;
  }

  void print(llvm::raw_ostream &os) const override;

private:
  std::size_t value;
};

class RelocationImmediate : public Immediate {
public:
  enum class RelocationFunction {
    Hi20,
    Lo12,
  };

  RelocationImmediate(RelocationFunction relocation, llvm::StringRef label)
      : Immediate(Kind::Relocation), relocation(relocation), label(label) {}

  RelocationFunction getRelocationFunction() const { return relocation; }
  llvm::StringRef getLabel() const { return label; }

  static bool classof(const Immediate *imm) {
    return imm->getKind() == Kind::Relocation;
  }

  void print(llvm::raw_ostream &os) const override;

private:
  RelocationFunction relocation;
  std::string label;
};

class ASMBuilder;

class DataSize {
public:
  enum class Kind {
    Byte,
    Half,
    Word,
    Double,
    SinglePrecision,
    DoublePrecision,
  };

private:
  Kind kind;
};

class Instruction {
public:
  virtual ~Instruction() = default;
  TypeID getId() const { return typeId; }

  virtual void print(llvm::raw_ostream &os) const;

protected:
  Instruction(TypeID id) : typeId(id) {}

private:
  friend class ASMBuilder;
  void setNode(Block::Node *node) { this->node = node; }
  Block::Node *node;
  TypeID typeId;
};

template <typename ConreteInst, typename ParentInst>
class InstructionTemplate : public ParentInst {
public:
  using ParentInst::ParentInst;
  using Base = InstructionTemplate<ConreteInst, ParentInst>;

  static bool classof(const Instruction *inst) {
    return inst->getId() == TypeID::get<ConreteInst>();
  }
};

class RType : public Instruction {
public:
  Register getRd() const { return rd; }
  Register getRs1() const { return rs1; }
  std::optional<Register> getRs2() const { return rs2; }

  static bool classof(const Instruction *inst);

protected:
  RType(TypeID id, Register rd, Register rs1, std::optional<Register> rs2)
      : Instruction(id), rd(rd), rs1(rs1), rs2(rs2) {}

private:
  Register rd;
  Register rs1;
  std::optional<Register> rs2;
};

namespace rtype {

class Add : public InstructionTemplate<Add, RType> {
public:
  DataSize getDataSize() const { return dataSize; }

private:
  friend class ASMBuilder;
  Add(Register rd, Register rs1, std::optional<Register> rs2,
      DataSize dataSize);
  DataSize dataSize;
};

class Sub : public InstructionTemplate<Sub, RType> {
public:
  DataSize getDataSize() const { return dataSize; }

private:
  friend class ASMBuilder;
  Sub(Register rd, Register rs1, std::optional<Register> rs2,
      DataSize dataSize);
  DataSize dataSize;
};

class Sll : public InstructionTemplate<Sll, RType> {
public:
  DataSize getDataSize() const { return dataSize; }

private:
  friend class ASMBuilder;
  Sll(Register rd, Register rs1, std::optional<Register> rs2,
      DataSize dataSize);
  DataSize dataSize;
};

class Srl : public InstructionTemplate<Srl, RType> {
public:
  DataSize getDataSize() const { return dataSize; }

private:
  friend class ASMBuilder;
  Srl(Register rd, Register rs1, std::optional<Register> rs2,
      DataSize dataSize);
  DataSize dataSize;
};

class Sra : public InstructionTemplate<Sra, RType> {
public:
  DataSize getDataSize() const { return dataSize; }

private:
  friend class ASMBuilder;
  Sra(Register rd, Register rs1, std::optional<Register> rs2,
      DataSize dataSize);
  DataSize dataSize;
};

class Mul : public InstructionTemplate<Mul, RType> {
public:
  DataSize getDataSize() const { return dataSize; }

private:
  friend class ASMBuilder;
  Mul(Register rd, Register rs1, std::optional<Register> rs2,
      DataSize dataSize);
  DataSize dataSize;
};

class Div : public InstructionTemplate<Div, RType> {
public:
  DataSize getDataSize() const { return dataSize; }
  bool isSigned() const { return isSignedV; }

private:
  friend class ASMBuilder;
  Div(Register rd, Register rs1, std::optional<Register> rs2, DataSize dataSize,
      bool isSigned);
  DataSize dataSize;
  bool isSignedV;
};

class Rem : public InstructionTemplate<Rem, RType> {
public:
  DataSize getDataSize() const { return dataSize; }
  bool isSigned() const { return isSignedV; }

private:
  friend class ASMBuilder;
  Rem(Register rd, Register rs1, std::optional<Register> rs2, DataSize dataSize,
      bool isSigned);
  DataSize dataSize;
  bool isSignedV;
};

class Slt : public InstructionTemplate<Slt, RType> {
public:
  bool isSigned() const { return isSignedV; }

private:
  friend class ASMBuilder;
  Slt(Register rd, Register rs1, std::optional<Register> rs2, bool isSigned);
  bool isSignedV;
};

class Xor : public InstructionTemplate<Xor, RType> {
public:
private:
  friend class ASMBuilder;
  Xor(Register rd, Register rs1, std::optional<Register> rs2);
};

class Or : public InstructionTemplate<Or, RType> {
public:
private:
  friend class ASMBuilder;
  Or(Register rd, Register rs1, std::optional<Register> rs2);
};

class And : public InstructionTemplate<And, RType> {
public:
private:
  friend class ASMBuilder;
  And(Register rd, Register rs1, std::optional<Register> rs2);
};

class Fadd : public InstructionTemplate<Fadd, RType> {
public:
  DataSize getDataSize() const { return dataSize; }

private:
  friend class ASMBuilder;
  Fadd(Register rd, Register rs1, std::optional<Register> rs2,
       DataSize dataSize);
  DataSize dataSize;
};

class Fsub : public InstructionTemplate<Fsub, RType> {
public:
  DataSize getDataSize() const { return dataSize; }

private:
  friend class ASMBuilder;
  Fsub(Register rd, Register rs1, std::optional<Register> rs2,
       DataSize dataSize);
  DataSize dataSize;
};

class Fmul : public InstructionTemplate<Fmul, RType> {
public:
  DataSize getDataSize() const { return dataSize; }

private:
  friend class ASMBuilder;
  Fmul(Register rd, Register rs1, std::optional<Register> rs2,
       DataSize dataSize);
  DataSize dataSize;
};

class Fdiv : public InstructionTemplate<Fdiv, RType> {
public:
  DataSize getDataSize() const { return dataSize; }

private:
  friend class ASMBuilder;
  Fdiv(Register rd, Register rs1, std::optional<Register> rs2,
       DataSize dataSize);
  DataSize dataSize;
};

class Feq : public InstructionTemplate<Feq, RType> {
public:
  DataSize getDataSize() const { return dataSize; }

private:
  friend class ASMBuilder;
  Feq(Register rd, Register rs1, std::optional<Register> rs2,
      DataSize dataSize);
  DataSize dataSize;
};

class Flt : public InstructionTemplate<Flt, RType> {
public:
  DataSize getDataSize() const { return dataSize; }

private:
  friend class ASMBuilder;
  Flt(Register rd, Register rs1, std::optional<Register> rs2,
      DataSize dataSize);
  DataSize dataSize;
};

class FmvIntToFloat : public InstructionTemplate<FmvIntToFloat, RType> {
public:
  DataSize getFloatDataSize() const { return floatDataSize; }

private:
  friend class ASMBuilder;
  FmvIntToFloat(Register rd, Register rs1, std::optional<Register> rs2,
                DataSize floatDataSize);
  DataSize floatDataSize;
};

class FmvFloatToInt : public InstructionTemplate<FmvFloatToInt, RType> {
public:
  DataSize getFloatDataSize() const { return floatDataSize; }

private:
  friend class ASMBuilder;
  FmvFloatToInt(Register rd, Register rs1, std::optional<Register> rs2,
                DataSize floatDataSize);
  DataSize floatDataSize;
};

class FcvtIntToFloat : public InstructionTemplate<FcvtIntToFloat, RType> {
public:
  DataSize getIntDataSize() const { return intDataSize; }
  DataSize getFloatDataSize() const { return floatDataSize; }
  bool isSigned() const { return isSignedV; }

private:
  friend class ASMBuilder;
  FcvtIntToFloat(Register rd, Register rs1, std::optional<Register> rs2,
                 DataSize intDataSize, DataSize floatDataSize, bool isSigned);
  DataSize intDataSize;
  DataSize floatDataSize;
  bool isSignedV;
};

class FcvtFloatToInt : public InstructionTemplate<FcvtFloatToInt, RType> {
public:
  DataSize getIntDataSize() const { return intDataSize; }
  DataSize getFloatDataSize() const { return floatDataSize; }
  bool isSigned() const { return isSignedV; }

private:
  friend class ASMBuilder;
  FcvtFloatToInt(Register rd, Register rs1, std::optional<Register> rs2,
                 DataSize floatDataSize, DataSize intDataSize, bool isSigned);
  DataSize floatDataSize;
  DataSize intDataSize;
  bool isSignedV;
};

class FcvtFloatToFloat : public InstructionTemplate<FcvtFloatToFloat, RType> {
public:
  DataSize getFromDataSize() const { return from; }
  DataSize getToDataSize() const { return to; }

private:
  friend class ASMBuilder;
  FcvtFloatToFloat(Register rd, Register rs1, std::optional<Register> rs2,
                   DataSize from, DataSize to);
  DataSize from;
  DataSize to;
};

}; // namespace rtype

class IType : public Instruction {
public:
  ~IType();
  Register getRd() const { return rd; }
  Register getRs1() const { return rs1; }
  Immediate *getImm() const { return imm; }

  static bool classof(const Instruction *inst);

protected:
  IType(TypeID id, Register rd, Register rs1, Immediate *imm)
      : Instruction(id), rd(rd), rs1(rs1), imm(imm) {}

private:
  Register rd;
  Register rs1;
  Immediate *imm;
};

namespace itype {

class Load : public InstructionTemplate<Load, IType> {
public:
  DataSize getDataSize() const { return dataSize; }
  bool isSigned() const { return isSignedV; }

private:
  friend class ASMBuilder;
  Load(Register rd, Register rs1, Immediate *imm, DataSize dataSize,
       bool isSigned);
  DataSize dataSize;
  bool isSignedV;
};

class Addi : public InstructionTemplate<Addi, IType> {
public:
  DataSize getDataSize() const { return dataSize; }

private:
  friend class ASMBuilder;
  Addi(Register rd, Register rs1, Immediate *imm, DataSize dataSize);
  DataSize dataSize;
};

class Xori : public InstructionTemplate<Xori, IType> {
public:
private:
  friend class ASMBuilder;
  using Base::Base;
};

class Ori : public InstructionTemplate<Ori, IType> {
public:
private:
  friend class ASMBuilder;
  using Base::Base;
};

class Andi : public InstructionTemplate<Andi, IType> {
public:
private:
  friend class ASMBuilder;
  using Base::Base;
};

class Slli : public InstructionTemplate<Slli, IType> {
public:
  DataSize getDataSize() const { return dataSize; }

private:
  friend class ASMBuilder;
  Slli(Register rd, Register rs1, Immediate *imm, DataSize dataSize);
  DataSize dataSize;
};

class Srli : public InstructionTemplate<Srli, IType> {
public:
  DataSize getDataSize() const { return dataSize; }

private:
  friend class ASMBuilder;
  Srli(Register rd, Register rs1, Immediate *imm, DataSize dataSize);
  DataSize dataSize;
};

class Srai : public InstructionTemplate<Srai, IType> {
public:
  DataSize getDataSize() const { return dataSize; }

private:
  friend class ASMBuilder;
  Srai(Register rd, Register rs1, Immediate *imm, DataSize dataSize);
  DataSize dataSize;
};

class Slti : public InstructionTemplate<Slti, IType> {
public:
  bool isSigned() const { return isSignedV; }

private:
  friend class ASMBuilder;
  Slti(Register rd, Register rs1, Immediate *imm, bool isSigned);
  bool isSignedV;
};

}; // namespace itype

class SType : public Instruction {
public:
  ~SType();
  Register getRs1() const { return rs1; }
  Register getRs2() const { return rs2; }
  Immediate *getImm() const { return imm; }

  static bool classof(const Instruction *inst);

protected:
  SType(TypeID id, Register rs1, Register rs2, Immediate *imm)
      : Instruction(id), rs1(rs1), rs2(rs2), imm(imm) {}

private:
  Register rs1;
  Register rs2;
  Immediate *imm;
};

namespace stype {
class Store : public InstructionTemplate<Store, SType> {
public:
  DataSize getDataSize() const { return dataSize; }

private:
  friend class ASMBuilder;
  Store(Register rs1, Register rs2, Immediate *imm, DataSize dataSize);
  DataSize dataSize;
};
} // namespace stype

class BType : public Instruction {
public:
  ~BType();
  Register getRs1() const { return rs1; }
  Register getRs2() const { return rs2; }
  Immediate *getImm() const { return imm; }

  static bool classof(const Instruction *inst);

protected:
  BType(TypeID id, Register rs1, Register rs2, Immediate *imm)
      : Instruction(id), rs1(rs1), rs2(rs2), imm(imm) {}

private:
  Register rs1;
  Register rs2;
  Immediate *imm;
};

namespace btype {

class Beq : public InstructionTemplate<Beq, BType> {
public:
private:
  friend class ASMBuilder;
  using Base::Base;
};

class Bne : public InstructionTemplate<Bne, BType> {
public:
private:
  friend class ASMBuilder;
  using Base::Base;
};

class Blt : public InstructionTemplate<Blt, BType> {
public:
  bool isSigned() const { return isSignedV; }

private:
  friend class ASMBuilder;
  Blt(Register rs1, Register rs2, Immediate *imm, bool isSigned);
  bool isSignedV;
};

class Bge : public InstructionTemplate<Bge, BType> {
public:
  bool isSigned() const { return isSignedV; }

private:
  friend class ASMBuilder;
  Bge(Register rs1, Register rs2, Immediate *imm, bool isSigned);
  bool isSignedV;
};

} // namespace btype

class UType : public Instruction {
public:
  ~UType();
  static bool classof(const Instruction *inst);

  Register getRd() const { return rd; }
  Immediate *getImm() const { return imm; }

protected:
  UType(TypeID id, Register rd, Immediate *imm)
      : Instruction(id), rd(rd), imm(imm) {}

private:
  Register rd;
  Immediate *imm;
};

namespace utype {
class Lui : public InstructionTemplate<Lui, UType> {
public:
private:
  friend class ASMBuilder;
  using Base::Base;
};
} // namespace utype

class Pseudo : public Instruction {
public:
  static bool classof(const Instruction *inst);

protected:
  using Instruction::Instruction;
};

namespace pseudo {

class La : public InstructionTemplate<La, Pseudo> {
public:
  ~La();
  Register getRd() const { return rd; }
  Immediate *getImm() const { return imm; }

private:
  friend class ASMBuilder;
  La(Register rd, Immediate *imm);
  Register rd;
  Immediate *imm;
};

class Li : public InstructionTemplate<Li, Pseudo> {
public:
  Register getRd() const { return rd; }
  std::size_t getImm() const { return imm; }

private:
  friend class ASMBuilder;
  Li(Register rd, std::size_t imm);
  Register rd;
  std::size_t imm;
};

class Mv : public InstructionTemplate<Mv, Pseudo> {
public:
  Register getRd() const { return rd; }
  Register getRs() const { return rs; }

private:
  friend class ASMBuilder;
  Mv(Register rd, Register rs);
  Register rd;
  Register rs;
};

class Fmv : public InstructionTemplate<Fmv, Pseudo> {
public:
  Register getRd() const { return rd; }
  Register getRs() const { return rs; }
  DataSize getDataSize() const { return dataSize; }

private:
  friend class ASMBuilder;
  Fmv(DataSize dataSize, Register rd, Register rs);
  DataSize dataSize;
  Register rd;
  Register rs;
};

class Neg : public InstructionTemplate<Neg, Pseudo> {
public:
  Register getRd() const { return rd; }
  Register getRs() const { return rs; }
  DataSize getDataSize() const { return dataSize; }

private:
  friend class ASMBuilder;
  Neg(DataSize dataSize, Register rd, Register rs);
  DataSize dataSize;
  Register rd;
  Register rs;
};

class SextW : public InstructionTemplate<SextW, Pseudo> {
public:
  Register getRd() const { return rd; }
  Register getRs() const { return rs; }

private:
  friend class ASMBuilder;
  SextW(Register rd, Register rs);
  Register rd;
  Register rs;
};

class Seqz : public InstructionTemplate<Seqz, Pseudo> {
public:
  Register getRd() const { return rd; }
  Register getRs() const { return rs; }

private:
  friend class ASMBuilder;
  Seqz(Register rd, Register rs);
  Register rd;
  Register rs;
};

class Snez : public InstructionTemplate<Snez, Pseudo> {
public:
  Register getRd() const { return rd; }
  Register getRs() const { return rs; }

private:
  friend class ASMBuilder;
  Snez(Register rd, Register rs);
  Register rd;
  Register rs;
};

class Fneg : public InstructionTemplate<Fneg, Pseudo> {
public:
  Register getRd() const { return rd; }
  Register getRs() const { return rs; }
  DataSize getDataSize() const { return dataSize; }

private:
  friend class ASMBuilder;
  Fneg(DataSize dataSize, Register rd, Register rs);
  DataSize dataSize;
  Register rd;
  Register rs;
};

class J : public InstructionTemplate<J, Pseudo> {
public:
  llvm::StringRef getLabel() const { return label; }

private:
  friend class ASMBuilder;
  J(llvm::StringRef label);
  std::string label;
};

class Jr : public InstructionTemplate<Jr, Pseudo> {
public:
  Register getRs() const { return rs; }

private:
  friend class ASMBuilder;
  Jr(Register rs);
  Register rs;
};

class Jalr : public InstructionTemplate<Jalr, Pseudo> {
public:
  Register getRs() const { return rs; }

private:
  friend class ASMBuilder;
  Jalr(Register rs);
  Register rs;
};

class Ret : public InstructionTemplate<Ret, Pseudo> {
public:
private:
  friend class ASMBuilder;
  using Base::Base;
};

class Call : public InstructionTemplate<Call, Pseudo> {
public:
  llvm::StringRef getLabel() const { return label; }

private:
  friend class ASMBuilder;
  Call(llvm::StringRef label);
  std::string label;
};

} // namespace pseudo

} // namespace kecc::as

DECLARE_KECC_TYPE_ID(kecc::as::rtype::Add)
DECLARE_KECC_TYPE_ID(kecc::as::rtype::Sub)
DECLARE_KECC_TYPE_ID(kecc::as::rtype::Sll)
DECLARE_KECC_TYPE_ID(kecc::as::rtype::Srl)
DECLARE_KECC_TYPE_ID(kecc::as::rtype::Sra)
DECLARE_KECC_TYPE_ID(kecc::as::rtype::Mul)
DECLARE_KECC_TYPE_ID(kecc::as::rtype::Div)
DECLARE_KECC_TYPE_ID(kecc::as::rtype::Rem)
DECLARE_KECC_TYPE_ID(kecc::as::rtype::Slt)
DECLARE_KECC_TYPE_ID(kecc::as::rtype::Xor)
DECLARE_KECC_TYPE_ID(kecc::as::rtype::Or)
DECLARE_KECC_TYPE_ID(kecc::as::rtype::And)
DECLARE_KECC_TYPE_ID(kecc::as::rtype::Fadd)
DECLARE_KECC_TYPE_ID(kecc::as::rtype::Fsub)
DECLARE_KECC_TYPE_ID(kecc::as::rtype::Fmul)
DECLARE_KECC_TYPE_ID(kecc::as::rtype::Fdiv)
DECLARE_KECC_TYPE_ID(kecc::as::rtype::Feq)
DECLARE_KECC_TYPE_ID(kecc::as::rtype::Flt)
DECLARE_KECC_TYPE_ID(kecc::as::rtype::FmvIntToFloat)
DECLARE_KECC_TYPE_ID(kecc::as::rtype::FmvFloatToInt)
DECLARE_KECC_TYPE_ID(kecc::as::rtype::FcvtIntToFloat)
DECLARE_KECC_TYPE_ID(kecc::as::rtype::FcvtFloatToInt)
DECLARE_KECC_TYPE_ID(kecc::as::rtype::FcvtFloatToFloat)

DECLARE_KECC_TYPE_ID(kecc::as::itype::Load)
DECLARE_KECC_TYPE_ID(kecc::as::itype::Addi)
DECLARE_KECC_TYPE_ID(kecc::as::itype::Xori)
DECLARE_KECC_TYPE_ID(kecc::as::itype::Ori)
DECLARE_KECC_TYPE_ID(kecc::as::itype::Andi)
DECLARE_KECC_TYPE_ID(kecc::as::itype::Slli)
DECLARE_KECC_TYPE_ID(kecc::as::itype::Srli)
DECLARE_KECC_TYPE_ID(kecc::as::itype::Srai)
DECLARE_KECC_TYPE_ID(kecc::as::itype::Slti)

DECLARE_KECC_TYPE_ID(kecc::as::stype::Store)

DECLARE_KECC_TYPE_ID(kecc::as::btype::Beq)
DECLARE_KECC_TYPE_ID(kecc::as::btype::Bne)
DECLARE_KECC_TYPE_ID(kecc::as::btype::Blt)
DECLARE_KECC_TYPE_ID(kecc::as::btype::Bge)

DECLARE_KECC_TYPE_ID(kecc::as::utype::Lui)

DECLARE_KECC_TYPE_ID(kecc::as::pseudo::La)
DECLARE_KECC_TYPE_ID(kecc::as::pseudo::Li)
DECLARE_KECC_TYPE_ID(kecc::as::pseudo::Mv)
DECLARE_KECC_TYPE_ID(kecc::as::pseudo::Fmv)
DECLARE_KECC_TYPE_ID(kecc::as::pseudo::Neg)
DECLARE_KECC_TYPE_ID(kecc::as::pseudo::SextW)
DECLARE_KECC_TYPE_ID(kecc::as::pseudo::Seqz)
DECLARE_KECC_TYPE_ID(kecc::as::pseudo::Snez)
DECLARE_KECC_TYPE_ID(kecc::as::pseudo::Fneg)
DECLARE_KECC_TYPE_ID(kecc::as::pseudo::J)
DECLARE_KECC_TYPE_ID(kecc::as::pseudo::Jr)
DECLARE_KECC_TYPE_ID(kecc::as::pseudo::Jalr)
DECLARE_KECC_TYPE_ID(kecc::as::pseudo::Ret)
DECLARE_KECC_TYPE_ID(kecc::as::pseudo::Call)

#endif // KECC_ASM_INSTRUCTION_H
