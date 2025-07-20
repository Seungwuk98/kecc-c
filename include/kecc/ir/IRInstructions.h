#ifndef KECC_IR_IR_INSTRUCTIONS_H
#define KECC_IR_IR_INSTRUCTIONS_H

#include "kecc/ir/Context.h"
#include "kecc/ir/IRAttributes.h"
#include "kecc/ir/IRBuilder.h"
#include "kecc/ir/Instruction.h"
#include "kecc/ir/JumpArg.h"

namespace kecc::ir::inst {

class Nop
    : public InstructionTemplate<Nop, Instruction, ZeroResult, OneResult> {
public:
  using Base::Base;
  static void build(IRBuilder &builder, InstructionState &state);

  static void printer(Nop op, IRPrintContext &context);
};

class Load : public InstructionTemplate<Load, Instruction, OneResult> {
public:
  using Base::Base;
  static void build(IRBuilder &builder, InstructionState &state, Value ptr);

  Value getPointer() const;

  static void printer(Load op, IRPrintContext &context);
};

class Store : public InstructionTemplate<Store, Instruction, NOperand<2>::Trait,
                                         OneResult, SideEffect> {
public:
  using Base::Base;
  static void build(IRBuilder &builder, InstructionState &state, Value value,
                    Value ptr);

  Value getValue() const;
  Value getPointer() const;

  static void printer(Store op, IRPrintContext &context);

private:
};

class Call : public InstructionTemplate<Call, Instruction, VariadicResults,
                                        SideEffect> {
public:
  using Base::Base;
  static void build(IRBuilder &builder, InstructionState &state, Value function,
                    llvm::ArrayRef<Value> args);

  Value getFunction() const;
  llvm::ArrayRef<Operand> getArguments() const;

  static void printer(Call op, IRPrintContext &context);

private:
};

class TypeCast : public InstructionTemplate<TypeCast, Instruction, OneResult> {
public:
  using Base::Base;
  static void build(IRBuilder &builder, InstructionState &state, Value value,
                    Type targetType);

  Value getValue() const;
  Type getTargetType() const;

  static void printer(TypeCast op, IRPrintContext &context);

private:
};

class Gep : public InstructionTemplate<Gep, Instruction, NOperand<2>::Trait,
                                       OneResult> {
public:
  using Base::Base;
  static void build(IRBuilder &builder, InstructionState &state, Value basePtr,
                    Value offset, Type type);

  Value getBasePointer() const;
  Value getOffset() const;

  static void printer(Gep op, IRPrintContext &context);
};

class Binary : public InstructionTemplate<Binary, Instruction,
                                          NOperand<2>::Trait, OneResult> {
public:
  using Base::Base;
  enum OpKind {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
  };

  static void build(IRBuilder &builder, InstructionState &state, Value lhs,
                    Value rhs, OpKind op);
  static void build(IRBuilder &builder, InstructionState &state, Value lhs,
                    Value rhs, OpKind op, Type result);

  Value getLhs() const;
  Value getRhs() const;
  OpKind getOpKind() const;

  static void printer(Binary op, IRPrintContext &context);

private:
};

class Unary : public InstructionTemplate<Unary, Instruction, OneResult> {
public:
  using Base::Base;
  enum OpKind {
    Plus,
    Minus,
    Negate,
  };

  static void build(IRBuilder &builder, InstructionState &state, Value value,
                    OpKind op);

  Value getValue() const;
  OpKind getOpKind() const;

  static void printer(Unary op, IRPrintContext &context);

private:
};

class Jump : public InstructionTemplate<Jump, BlockExit, ZeroResult> {
public:
  using Base::Base;
  static void build(IRBuilder &builder, InstructionState &state,
                    JumpArgState arg);

  JumpArg *getJumpArg() const;

  static void printer(Jump op, IRPrintContext &context);
};

class Branch : public InstructionTemplate<Branch, BlockExit, ZeroResult> {
public:
  using Base::Base;
  static void build(IRBuilder &builder, InstructionState &state, Value value,
                    JumpArgState ifArg, JumpArgState elseArg);

  Value getCondition() const;
  JumpArg *getIfArg() const;
  JumpArg *getElseArg() const;

  static void printer(Branch op, IRPrintContext &context);
};

class Switch : public InstructionTemplate<Switch, BlockExit, ZeroResult> {
public:
  using Base::Base;
  static void build(IRBuilder &builder, InstructionState &state, Value value,
                    llvm::ArrayRef<Value> caseValues,
                    llvm::ArrayRef<JumpArgState> cases,
                    JumpArgState defaultCase);

  Value getValue() const;
  Value getCaseValue(size_t idx) const;
  JumpArg *getCaseJumpArg(size_t idx) const;
  std::size_t getCaseSize() const;
  JumpArg *getDefaultCase() const;

  static void printer(Switch op, IRPrintContext &context);
};

class Return : public InstructionTemplate<Return, BlockExit, ZeroResult> {
public:
  using Base::Base;

  static void build(IRBuilder &builder, InstructionState &state);
  static void build(IRBuilder &builder, InstructionState &state, Value value);
  static void build(IRBuilder &builder, InstructionState &state,
                    llvm::ArrayRef<Value> value);

  Value getValue(std::size_t idx) const;
  llvm::ArrayRef<Operand> getValues() const;
  std::size_t getValueSize() const { return getValues().size(); }

  static void printer(Return op, IRPrintContext &context);
};

class Unreachable
    : public InstructionTemplate<Unreachable, BlockExit, ZeroResult> {
public:
  using Base::Base;
  static void build(IRBuilder &builder, InstructionState &state);

  static void printer(Unreachable op, IRPrintContext &context);
};

class Constant : public InstructionTemplate<Constant, Instruction, OneResult> {
public:
  using Base::Base;

  static void build(IRBuilder &builder, InstructionState &state,
                    ConstantAttr value);

  static void printer(Constant op, IRPrintContext &context);

  ConstantAttr getValue() const;

  void replaceValue(ConstantAttr value);
};

class StructDefinition
    : public InstructionTemplate<StructDefinition, Instruction> {
public:
  using Base::Base;
  static void build(IRBuilder &builder, InstructionState &state,
                    llvm::ArrayRef<std::pair<llvm::StringRef, Type>> fields,
                    llvm::StringRef name);

  std::pair<llvm::StringRef, Type> getField(size_t idx) const;
  size_t getFieldSize() const;
  llvm::StringRef getName() const;

  static void printer(StructDefinition op, IRPrintContext &context);
};

class GlobalVariableDefinition
    : public InstructionTemplate<GlobalVariableDefinition, Instruction> {
public:
  using Base::Base;
  static void build(IRBuilder &builder, InstructionState &state, Type type,
                    llvm::StringRef name);
  static void build(IRBuilder &builder, InstructionState &state, Type type,
                    llvm::StringRef name, InitializerAttr initializer);

  llvm::StringRef getName() const;
  Attribute getInitializer() const;
  Type getType() const;
  void interpretInitializer();

  static void printer(GlobalVariableDefinition op, IRPrintContext &context);
};

class LocalVariable
    : public InstructionTemplate<LocalVariable, Instruction, OneResult> {
public:
  using Base::Base;
  static void build(IRBuilder &builder, InstructionState &state, Type type);

  static bool verify(LocalVariable op);

  void printAsDef(IRPrintContext &context) const;

  static void printer(LocalVariable op, IRPrintContext &context);
};

class Unresolved
    : public InstructionTemplate<Unresolved, Instruction, OneResult> {
public:
  using Base::Base;
  static void build(IRBuilder &builder, InstructionState &state, Type type);

  static void printer(Unresolved op, IRPrintContext &context);
};

} // namespace kecc::ir::inst

#endif // KECC_IR_IR_INSTRUCTIONS_H
