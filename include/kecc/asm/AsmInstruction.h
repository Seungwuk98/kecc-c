#ifndef KECC_ASM_INSTRUCTION_H
#define KECC_ASM_INSTRUCTION_H

#include "kecc/asm/Asm.h"
#include "kecc/asm/Register.h"
#include "kecc/ir/Type.h"
#include "kecc/utils/MLIR.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

namespace kecc::as {

template <typename T> struct PointerCastBase {
  template <typename... Us> bool isa() const {
    return llvm::isa<Us...>(derived());
  }
  template <typename U> U *cast() { return llvm::cast<U>(derived()); }
  template <typename U> const U *cast() const {
    return llvm::cast<U>(derived());
  }
  template <typename U> U *dyn_cast() { return llvm::dyn_cast<U>(derived()); }
  template <typename U> const U *dyn_cast() const {
    return llvm::dyn_cast<U>(derived());
  }

private:
  T *derived() { return static_cast<T *>(this); }
  const T *derived() const { return static_cast<const T *>(this); }
};

class Immediate : public PointerCastBase<Immediate> {
public:
  virtual ~Immediate() = default;
  enum class Kind {
    Value,
    Relocation,
  };

  Kind getKind() const { return kind; }

  virtual void print(llvm::raw_ostream &os) const = 0;
  std::string toString() const;

protected:
  Immediate(Kind kind) : kind(kind) {}

private:
  Kind kind;
};

class ValueImmediate final : public Immediate {
public:
  ValueImmediate(std::size_t value) : Immediate(Kind::Value), value(value) {}

  std::int64_t getValue() const { return value; }

  static bool classof(const Immediate *imm) {
    return imm->getKind() == Kind::Value;
  }

  void print(llvm::raw_ostream &os) const override;

private:
  std::int64_t value;
};

class RelocationImmediate final : public Immediate {
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

class AsmBuilder;

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

  DataSize(Kind kind) : kind(kind) {}

  static DataSize byte() { return DataSize(Kind::Byte); }
  static DataSize half() { return DataSize(Kind::Half); }
  static DataSize word() { return DataSize(Kind::Word); }
  static DataSize doubleWord() { return DataSize(Kind::Double); }
  static DataSize singlePrecision() { return DataSize(Kind::SinglePrecision); }
  static DataSize doublePrecision() { return DataSize(Kind::DoublePrecision); }

  std::string toString() const {
    switch (kind) {
    case Kind::Byte:
      return "b";
    case Kind::Half:
      return "h";
    case Kind::Word:
      return "w";
    case Kind::Double:
      return "d";
    case Kind::SinglePrecision:
      return "s";
    case Kind::DoublePrecision:
      return "d";
    }
  }

  bool isByte() const { return kind == Kind::Byte; }
  bool isHalf() const { return kind == Kind::Half; }
  bool isWord() const { return kind == Kind::Word; }
  bool isDouble() const { return kind == Kind::Double; }
  bool isSinglePrecision() const { return kind == Kind::SinglePrecision; }
  bool isDoublePrecision() const { return kind == Kind::DoublePrecision; }
  bool isFloat() const {
    return kind == Kind::SinglePrecision || kind == Kind::DoublePrecision;
  }
  bool isInt() const {
    return kind == Kind::Byte || kind == Kind::Half || kind == Kind::Word ||
           kind == Kind::Double;
  }

  static DataSize tryFrom(ir::Type type);

private:
  Kind kind;
};

class Instruction : public PointerCastBase<Instruction> {
public:
  virtual ~Instruction() = default;
  TypeID getId() const { return typeId; }
  Block *getParentBlock() const { return parent; }
  Block::Node *getNode() const { return node; }

  virtual void print(llvm::raw_ostream &os) const = 0;

protected:
  Instruction(TypeID id) : typeId(id) {}

private:
  friend class AsmBuilder;
  void setNode(Block::Node *node) { this->node = node; }
  void setParent(Block *parent) { this->parent = parent; }
  Block *parent;
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

  void print(llvm::raw_ostream &os) const override final;

  virtual std::string toString() const = 0;

protected:
  RType(TypeID id, Register rd, Register rs1, std::optional<Register> rs2)
      : Instruction(id), rd(rd), rs1(rs1), rs2(rs2) {}

private:
  Register rd;
  Register rs1;
  std::optional<Register> rs2;
};

namespace rtype {

class Add final : public InstructionTemplate<Add, RType> {
public:
  DataSize getDataSize() const { return dataSize; }

  std::string toString() const override;

private:
  friend class AsmBuilder;
  Add(Register rd, Register rs1, std::optional<Register> rs2,
      DataSize dataSize);
  DataSize dataSize;
};

class Sub final : public InstructionTemplate<Sub, RType> {
public:
  DataSize getDataSize() const { return dataSize; }

  std::string toString() const override;

private:
  friend class AsmBuilder;
  Sub(Register rd, Register rs1, std::optional<Register> rs2,
      DataSize dataSize);
  DataSize dataSize;
};

class Sll final : public InstructionTemplate<Sll, RType> {
public:
  DataSize getDataSize() const { return dataSize; }

  std::string toString() const override;

private:
  friend class AsmBuilder;
  Sll(Register rd, Register rs1, std::optional<Register> rs2,
      DataSize dataSize);
  DataSize dataSize;
};

class Srl final : public InstructionTemplate<Srl, RType> {
public:
  DataSize getDataSize() const { return dataSize; }

  std::string toString() const override;

private:
  friend class AsmBuilder;
  Srl(Register rd, Register rs1, std::optional<Register> rs2,
      DataSize dataSize);
  DataSize dataSize;
};

class Sra final : public InstructionTemplate<Sra, RType> {
public:
  DataSize getDataSize() const { return dataSize; }

  std::string toString() const override;

private:
  friend class AsmBuilder;
  Sra(Register rd, Register rs1, std::optional<Register> rs2,
      DataSize dataSize);
  DataSize dataSize;
};

class Mul final : public InstructionTemplate<Mul, RType> {
public:
  DataSize getDataSize() const { return dataSize; }

  std::string toString() const override;

private:
  friend class AsmBuilder;
  Mul(Register rd, Register rs1, std::optional<Register> rs2,
      DataSize dataSize);
  DataSize dataSize;
};

class Div final : public InstructionTemplate<Div, RType> {
public:
  DataSize getDataSize() const { return dataSize; }
  bool isSigned() const { return isSignedV; }
  std::string toString() const override;

private:
  friend class AsmBuilder;
  Div(Register rd, Register rs1, std::optional<Register> rs2, DataSize dataSize,
      bool isSigned);
  DataSize dataSize;
  bool isSignedV;
};

class Rem final : public InstructionTemplate<Rem, RType> {
public:
  DataSize getDataSize() const { return dataSize; }
  bool isSigned() const { return isSignedV; }

  std::string toString() const override;

private:
  friend class AsmBuilder;
  Rem(Register rd, Register rs1, std::optional<Register> rs2, DataSize dataSize,
      bool isSigned);
  DataSize dataSize;
  bool isSignedV;
};

class Slt final : public InstructionTemplate<Slt, RType> {
public:
  bool isSigned() const { return isSignedV; }

  std::string toString() const override;

private:
  friend class AsmBuilder;
  Slt(Register rd, Register rs1, std::optional<Register> rs2, bool isSigned);
  bool isSignedV;
};

class Xor final : public InstructionTemplate<Xor, RType> {
public:
  std::string toString() const override;

private:
  friend class AsmBuilder;
  Xor(Register rd, Register rs1, std::optional<Register> rs2);
};

class Or final : public InstructionTemplate<Or, RType> {
public:
  std::string toString() const override;

private:
  friend class AsmBuilder;
  Or(Register rd, Register rs1, std::optional<Register> rs2);
};

class And final : public InstructionTemplate<And, RType> {
public:
  std::string toString() const override;

private:
  friend class AsmBuilder;
  And(Register rd, Register rs1, std::optional<Register> rs2);
};

class Fadd final : public InstructionTemplate<Fadd, RType> {
public:
  DataSize getDataSize() const { return dataSize; }

  std::string toString() const override;

private:
  friend class AsmBuilder;
  Fadd(Register rd, Register rs1, std::optional<Register> rs2,
       DataSize dataSize);
  DataSize dataSize;
};

class Fsub final : public InstructionTemplate<Fsub, RType> {
public:
  DataSize getDataSize() const { return dataSize; }

  std::string toString() const override;

private:
  friend class AsmBuilder;
  Fsub(Register rd, Register rs1, std::optional<Register> rs2,
       DataSize dataSize);
  DataSize dataSize;
};

class Fmul final : public InstructionTemplate<Fmul, RType> {
public:
  DataSize getDataSize() const { return dataSize; }

  std::string toString() const override;

private:
  friend class AsmBuilder;
  Fmul(Register rd, Register rs1, std::optional<Register> rs2,
       DataSize dataSize);
  DataSize dataSize;
};

class Fdiv final : public InstructionTemplate<Fdiv, RType> {
public:
  DataSize getDataSize() const { return dataSize; }

  std::string toString() const override;

private:
  friend class AsmBuilder;
  Fdiv(Register rd, Register rs1, std::optional<Register> rs2,
       DataSize dataSize);
  DataSize dataSize;
};

class Feq final : public InstructionTemplate<Feq, RType> {
public:
  DataSize getDataSize() const { return dataSize; }

  std::string toString() const override;

private:
  friend class AsmBuilder;
  Feq(Register rd, Register rs1, std::optional<Register> rs2,
      DataSize dataSize);
  DataSize dataSize;
};

class Flt final : public InstructionTemplate<Flt, RType> {
public:
  DataSize getDataSize() const { return dataSize; }

  std::string toString() const override;

private:
  friend class AsmBuilder;
  Flt(Register rd, Register rs1, std::optional<Register> rs2,
      DataSize dataSize);
  DataSize dataSize;
};

class FmvIntToFloat final : public InstructionTemplate<FmvIntToFloat, RType> {
public:
  DataSize getFloatDataSize() const { return floatDataSize; }

  std::string toString() const override;

private:
  friend class AsmBuilder;
  FmvIntToFloat(Register rd, Register rs1, std::optional<Register> rs2,
                DataSize floatDataSize);
  DataSize floatDataSize;
};

class FmvFloatToInt final : public InstructionTemplate<FmvFloatToInt, RType> {
public:
  DataSize getFloatDataSize() const { return floatDataSize; }

  std::string toString() const override;

private:
  friend class AsmBuilder;
  FmvFloatToInt(Register rd, Register rs1, std::optional<Register> rs2,
                DataSize floatDataSize);
  DataSize floatDataSize;
};

class FcvtIntToFloat final : public InstructionTemplate<FcvtIntToFloat, RType> {
public:
  DataSize getIntDataSize() const { return intDataSize; }
  DataSize getFloatDataSize() const { return floatDataSize; }
  bool isSigned() const { return isSignedV; }

  std::string toString() const override;

private:
  friend class AsmBuilder;
  FcvtIntToFloat(Register rd, Register rs1, std::optional<Register> rs2,
                 DataSize intDataSize, DataSize floatDataSize, bool isSigned);
  DataSize intDataSize;
  DataSize floatDataSize;
  bool isSignedV;
};

class FcvtFloatToInt final : public InstructionTemplate<FcvtFloatToInt, RType> {
public:
  DataSize getIntDataSize() const { return intDataSize; }
  DataSize getFloatDataSize() const { return floatDataSize; }
  bool isSigned() const { return isSignedV; }

  std::string toString() const override;

private:
  friend class AsmBuilder;
  FcvtFloatToInt(Register rd, Register rs1, std::optional<Register> rs2,
                 DataSize floatDataSize, DataSize intDataSize, bool isSigned);
  DataSize floatDataSize;
  DataSize intDataSize;
  bool isSignedV;
};

class FcvtFloatToFloat final
    : public InstructionTemplate<FcvtFloatToFloat, RType> {
public:
  DataSize getFromDataSize() const { return from; }
  DataSize getToDataSize() const { return to; }

  std::string toString() const override;

private:
  friend class AsmBuilder;
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

  void print(llvm::raw_ostream &os) const override final;

  virtual std::string toString() const = 0;

protected:
  IType(TypeID id, Register rd, Register rs1, Immediate *imm)
      : Instruction(id), rd(rd), rs1(rs1), imm(imm) {}

private:
  Register rd;
  Register rs1;
  Immediate *imm;
};

namespace itype {

class Load final : public InstructionTemplate<Load, IType> {
public:
  DataSize getDataSize() const { return dataSize; }
  bool isSigned() const { return isSignedV; }

  std::string toString() const override;

private:
  friend class AsmBuilder;
  Load(Register rd, Register rs1, Immediate *imm, DataSize dataSize,
       bool isSigned);
  DataSize dataSize;
  bool isSignedV;
};

class Addi final : public InstructionTemplate<Addi, IType> {
public:
  DataSize getDataSize() const { return dataSize; }

  std::string toString() const override;

private:
  friend class AsmBuilder;
  Addi(Register rd, Register rs1, Immediate *imm, DataSize dataSize);
  DataSize dataSize;
};

class Xori final : public InstructionTemplate<Xori, IType> {
public:
  std::string toString() const override;

private:
  friend class AsmBuilder;
  using Base::Base;
};

class Ori final : public InstructionTemplate<Ori, IType> {
public:
  std::string toString() const override;

private:
  friend class AsmBuilder;
  using Base::Base;
};

class Andi final : public InstructionTemplate<Andi, IType> {
public:
  std::string toString() const override;

private:
  friend class AsmBuilder;
  using Base::Base;
};

class Slli final : public InstructionTemplate<Slli, IType> {
public:
  DataSize getDataSize() const { return dataSize; }

  std::string toString() const override;

private:
  friend class AsmBuilder;
  Slli(Register rd, Register rs1, Immediate *imm, DataSize dataSize);
  DataSize dataSize;
};

class Srli final : public InstructionTemplate<Srli, IType> {
public:
  DataSize getDataSize() const { return dataSize; }

  std::string toString() const override;

private:
  friend class AsmBuilder;
  Srli(Register rd, Register rs1, Immediate *imm, DataSize dataSize);
  DataSize dataSize;
};

class Srai final : public InstructionTemplate<Srai, IType> {
public:
  DataSize getDataSize() const { return dataSize; }

  std::string toString() const override;

private:
  friend class AsmBuilder;
  Srai(Register rd, Register rs1, Immediate *imm, DataSize dataSize);
  DataSize dataSize;
};

class Slti final : public InstructionTemplate<Slti, IType> {
public:
  bool isSigned() const { return isSignedV; }

  std::string toString() const override;

private:
  friend class AsmBuilder;
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

  void print(llvm::raw_ostream &os) const override final;

  virtual std::string toString() const = 0;

protected:
  SType(TypeID id, Register rs1, Register rs2, Immediate *imm)
      : Instruction(id), rs1(rs1), rs2(rs2), imm(imm) {}

private:
  Register rs1;
  Register rs2;
  Immediate *imm;
};

namespace stype {
class Store final : public InstructionTemplate<Store, SType> {
public:
  DataSize getDataSize() const { return dataSize; }

  std::string toString() const override;

private:
  friend class AsmBuilder;
  Store(Register rs1, Register rs2, Immediate *imm, DataSize dataSize);
  DataSize dataSize;
};
} // namespace stype

class BType : public Instruction {
public:
  Register getRs1() const { return rs1; }
  Register getRs2() const { return rs2; }
  llvm::StringRef getImm() const { return imm; }

  static bool classof(const Instruction *inst);

  void print(llvm::raw_ostream &os) const override final;

  virtual std::string toString() const = 0;

protected:
  BType(TypeID id, Register rs1, Register rs2, llvm::StringRef imm)
      : Instruction(id), rs1(rs1), rs2(rs2), imm(imm) {}

private:
  Register rs1;
  Register rs2;
  std::string imm;
};

namespace btype {

class Beq final : public InstructionTemplate<Beq, BType> {
public:
  std::string toString() const override;

private:
  friend class AsmBuilder;
  using Base::Base;
};

class Bne final : public InstructionTemplate<Bne, BType> {
public:
  std::string toString() const override;

private:
  friend class AsmBuilder;
  using Base::Base;
};

class Blt final : public InstructionTemplate<Blt, BType> {
public:
  bool isSigned() const { return isSignedV; }
  std::string toString() const override;

private:
  friend class AsmBuilder;
  Blt(Register rs1, Register rs2, llvm::StringRef imm, bool isSigned);
  bool isSignedV;
};

class Bge final : public InstructionTemplate<Bge, BType> {
public:
  bool isSigned() const { return isSignedV; }

  std::string toString() const override;

private:
  friend class AsmBuilder;
  Bge(Register rs1, Register rs2, llvm::StringRef imm, bool isSigned);
  bool isSignedV;
};

} // namespace btype

class UType : public Instruction {
public:
  ~UType();
  static bool classof(const Instruction *inst);

  Register getRd() const { return rd; }
  Immediate *getImm() const { return imm; }

  void print(llvm::raw_ostream &os) const override final;

  virtual std::string toString() const = 0;

protected:
  UType(TypeID id, Register rd, Immediate *imm)
      : Instruction(id), rd(rd), imm(imm) {}

private:
  Register rd;
  Immediate *imm;
};

namespace utype {
class Lui final : public InstructionTemplate<Lui, UType> {
public:
  std::string toString() const override;

private:
  friend class AsmBuilder;
  using Base::Base;
};
} // namespace utype

class Pseudo : public Instruction {
public:
  static bool classof(const Instruction *inst);

  void print(llvm::raw_ostream &os) const override final;

  virtual std::string toString() const = 0;

protected:
  using Instruction::Instruction;
};

namespace pseudo {

class La final : public InstructionTemplate<La, Pseudo> {
public:
  ~La();
  Register getRd() const { return rd; }
  llvm::StringRef getSymbol() const { return symbol; }

  std::string toString() const override;

private:
  friend class AsmBuilder;
  La(Register rd, llvm::StringRef symbol);
  Register rd;
  std::string symbol;
};

class Li final : public InstructionTemplate<Li, Pseudo> {
public:
  Register getRd() const { return rd; }
  std::size_t getImm() const { return imm; }

  std::string toString() const override;

private:
  friend class AsmBuilder;
  Li(Register rd, std::size_t imm);
  Register rd;
  std::size_t imm;
};

class Mv final : public InstructionTemplate<Mv, Pseudo> {
public:
  Register getRd() const { return rd; }
  Register getRs() const { return rs; }

  std::string toString() const override;

private:
  friend class AsmBuilder;
  Mv(Register rd, Register rs);
  Register rd;
  Register rs;
};

class Fmv final : public InstructionTemplate<Fmv, Pseudo> {
public:
  Register getRd() const { return rd; }
  Register getRs() const { return rs; }
  DataSize getDataSize() const { return dataSize; }

  std::string toString() const override;

private:
  friend class AsmBuilder;
  Fmv(DataSize dataSize, Register rd, Register rs);
  DataSize dataSize;
  Register rd;
  Register rs;
};

class Neg final : public InstructionTemplate<Neg, Pseudo> {
public:
  Register getRd() const { return rd; }
  Register getRs() const { return rs; }
  DataSize getDataSize() const { return dataSize; }

  std::string toString() const override;

private:
  friend class AsmBuilder;
  Neg(DataSize dataSize, Register rd, Register rs);
  DataSize dataSize;
  Register rd;
  Register rs;
};

class SextW final : public InstructionTemplate<SextW, Pseudo> {
public:
  Register getRd() const { return rd; }
  Register getRs() const { return rs; }

  std::string toString() const override;

private:
  friend class AsmBuilder;
  SextW(Register rd, Register rs);
  Register rd;
  Register rs;
};

class Seqz final : public InstructionTemplate<Seqz, Pseudo> {
public:
  Register getRd() const { return rd; }
  Register getRs() const { return rs; }

  std::string toString() const override;

private:
  friend class AsmBuilder;
  Seqz(Register rd, Register rs);
  Register rd;
  Register rs;
};

class Snez final : public InstructionTemplate<Snez, Pseudo> {
public:
  Register getRd() const { return rd; }
  Register getRs() const { return rs; }

  std::string toString() const override;

private:
  friend class AsmBuilder;
  Snez(Register rd, Register rs);
  Register rd;
  Register rs;
};

class Fneg final : public InstructionTemplate<Fneg, Pseudo> {
public:
  Register getRd() const { return rd; }
  Register getRs() const { return rs; }
  DataSize getDataSize() const { return dataSize; }

  std::string toString() const override;

private:
  friend class AsmBuilder;
  Fneg(DataSize dataSize, Register rd, Register rs);
  DataSize dataSize;
  Register rd;
  Register rs;
};

class J final : public InstructionTemplate<J, Pseudo> {
public:
  llvm::StringRef getLabel() const { return label; }

  std::string toString() const override;

private:
  friend class AsmBuilder;
  J(llvm::StringRef label);
  std::string label;
};

class Jr final : public InstructionTemplate<Jr, Pseudo> {
public:
  Register getRs() const { return rs; }

  std::string toString() const override;

private:
  friend class AsmBuilder;
  Jr(Register rs);
  Register rs;
};

class Jalr final : public InstructionTemplate<Jalr, Pseudo> {
public:
  Register getRs() const { return rs; }

  std::string toString() const override;

private:
  friend class AsmBuilder;
  Jalr(Register rs);
  Register rs;
};

class Ret final : public InstructionTemplate<Ret, Pseudo> {
public:
  std::string toString() const override;

private:
  friend class AsmBuilder;
  using Base::Base;
};

class Call final : public InstructionTemplate<Call, Pseudo> {
public:
  llvm::StringRef getLabel() const { return label; }

  std::string toString() const override;

private:
  friend class AsmBuilder;
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
