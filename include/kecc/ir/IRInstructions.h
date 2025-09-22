#ifndef KECC_IR_IR_INSTRUCTIONS_H
#define KECC_IR_IR_INSTRUCTIONS_H

#include "kecc/ir/Context.h"
#include "kecc/ir/IRAttributes.h"
#include "kecc/ir/IRBuilder.h"
#include "kecc/ir/IRTransforms.h"
#include "kecc/ir/Instruction.h"
#include "kecc/ir/JumpArg.h"

namespace kecc::ir::inst {

class Nop : public InstructionTemplate<Nop, Instruction, ZeroResult, OneResult,
                                       Pure> {
public:
  using Base::Base;
  static void build(IRBuilder &builder, InstructionState &state);

  static void printer(Nop op, IRPrintContext &context);
  static llvm::StringRef getDebugName() { return "nop"; }

  struct Adaptor {
    Adaptor(llvm::ArrayRef<Value>, llvm::ArrayRef<JumpArgState>) {}
  };
};

class Load
    : public InstructionTemplate<Load, Instruction, OneResult, ReadMemory> {
public:
  using Base::Base;
  static void build(IRBuilder &builder, InstructionState &state, Value ptr);

  Value getPointer() const;
  const Operand &getPointerAsOperand() const;

  static void printer(Load op, IRPrintContext &context);
  static llvm::StringRef getDebugName() { return "load"; }

  struct Adaptor {
    Adaptor(llvm::ArrayRef<Value> operands, llvm::ArrayRef<JumpArgState>)
        : operands(operands) {}
    Value getPointer() const;
    llvm::ArrayRef<Value> operands;
  };
};

class Store : public InstructionTemplate<Store, Instruction, NOperand<2>::Trait,
                                         OneResult, SideEffect, WriteMemory> {
public:
  using Base::Base;
  static void build(IRBuilder &builder, InstructionState &state, Value value,
                    Value ptr);

  Value getValue() const;
  const Operand &getValueAsOperand() const;
  Value getPointer() const;
  const Operand &getPointerAsOperand() const;

  static void printer(Store op, IRPrintContext &context);
  static llvm::StringRef getDebugName() { return "store"; }
  struct Adaptor {
    Adaptor(llvm::ArrayRef<Value> operands, llvm::ArrayRef<JumpArgState>)
        : operands(operands) {}

    Value getValue() const;
    Value getPointer() const;

    llvm::ArrayRef<Value> operands;
  };
};

class Call : public InstructionTemplate<Call, Instruction, VariadicResults,
                                        SideEffect, CallLike> {
public:
  using Base::Base;
  static void build(IRBuilder &builder, InstructionState &state, Value function,
                    llvm::ArrayRef<Value> args);
  static void build(IRBuilder &builder, InstructionState &state, Value function,
                    llvm::ArrayRef<Value> args, llvm::ArrayRef<Type> types);

  Value getFunction() const;
  const Operand &getFunctionAsOperand() const;
  llvm::ArrayRef<Operand> getArguments() const;

  static void printer(Call op, IRPrintContext &context);
  static llvm::StringRef getDebugName() { return "call"; }

  struct Adaptor {
    Adaptor(llvm::ArrayRef<Value> operands, llvm::ArrayRef<JumpArgState>)
        : operands(operands) {}

    Value getFunction() const;
    llvm::ArrayRef<Value> getArguments() const;

    llvm::ArrayRef<Value> operands;
  };
};

class TypeCast
    : public InstructionTemplate<TypeCast, Instruction, OneResult, Pure> {
public:
  using Base::Base;
  static void build(IRBuilder &builder, InstructionState &state, Value value,
                    Type targetType);

  Value getValue() const;
  const Operand &getValueAsOperand() const;
  Type getTargetType() const;

  static void printer(TypeCast op, IRPrintContext &context);
  static llvm::StringRef getDebugName() { return "typecast"; }

  struct Adaptor {
    Adaptor(llvm::ArrayRef<Value> operands, llvm::ArrayRef<JumpArgState>)
        : operands(operands) {}

    Value getValue() const;

    llvm::ArrayRef<Value> operands;
  };
};

class Gep : public InstructionTemplate<Gep, Instruction, NOperand<2>::Trait,
                                       OneResult, Pure> {
public:
  using Base::Base;
  static void build(IRBuilder &builder, InstructionState &state, Value basePtr,
                    Value offset, Type type);

  Value getBasePointer() const;
  const Operand &getBasePointerAsOperand() const;
  Value getOffset() const;
  const Operand &getOffsetAsOperand() const;

  static void printer(Gep op, IRPrintContext &context);
  static llvm::StringRef getDebugName() { return "getelementptr"; }

  struct Adaptor {
    Adaptor(llvm::ArrayRef<Value> operands, llvm::ArrayRef<JumpArgState>)
        : operands(operands) {}

    Value getBasePointer() const;
    Value getOffset() const;

    llvm::ArrayRef<Value> operands;
  };
};

class Binary : public InstructionTemplate<Binary, Instruction,
                                          NOperand<2>::Trait, OneResult, Pure> {
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
  const Operand &getLhsAsOperand() const;
  Value getRhs() const;
  const Operand &getRhsAsOperand() const;
  OpKind getOpKind() const;

  static void printer(Binary op, IRPrintContext &context);
  static llvm::StringRef getDebugName() { return "binary"; }

  struct Adaptor {
    Adaptor(llvm::ArrayRef<Value> operands, llvm::ArrayRef<JumpArgState>)
        : operands(operands) {}

    Value getLhs() const;
    Value getRhs() const;

    llvm::ArrayRef<Value> operands;
  };
};

class Unary : public InstructionTemplate<Unary, Instruction, OneResult, Pure> {
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
  const Operand &getValueAsOperand() const;
  OpKind getOpKind() const;

  static void printer(Unary op, IRPrintContext &context);
  static llvm::StringRef getDebugName() { return "unary"; }

  struct Adaptor {
    Adaptor(llvm::ArrayRef<Value> operands, llvm::ArrayRef<JumpArgState>)
        : operands(operands) {}

    Value getValue() const;

    llvm::ArrayRef<Value> operands;
  };
};

class Jump : public InstructionTemplate<Jump, BlockExit, ZeroResult> {
public:
  using Base::Base;
  static void build(IRBuilder &builder, InstructionState &state,
                    JumpArgState arg);

  JumpArg *getJumpArg() const;

  static void printer(Jump op, IRPrintContext &context);
  static llvm::StringRef getDebugName() { return "jump"; }

  struct Adaptor {
    Adaptor(llvm::ArrayRef<Value>, llvm::ArrayRef<JumpArgState> jumpArgs)
        : jumpArgs(jumpArgs) {}
    const JumpArgState &getJumpArg() const;
    llvm::ArrayRef<JumpArgState> jumpArgs;
  };
};

class Branch : public InstructionTemplate<Branch, BlockExit, ZeroResult> {
public:
  using Base::Base;
  static void build(IRBuilder &builder, InstructionState &state, Value value,
                    JumpArgState ifArg, JumpArgState elseArg);

  Value getCondition() const;
  const Operand &getConditionAsOperand() const;
  JumpArg *getIfArg() const;
  JumpArg *getElseArg() const;

  static void printer(Branch op, IRPrintContext &context);
  static llvm::StringRef getDebugName() { return "branch"; }

  struct Adaptor {
    Adaptor(llvm::ArrayRef<Value> operands,
            llvm::ArrayRef<JumpArgState> jumpArgs)
        : operands(operands), jumpArgs(jumpArgs) {}

    Value getCondition() const;
    JumpArgState getIfArg() const;
    JumpArgState getElseArg() const;

    llvm::ArrayRef<Value> operands;
    llvm::ArrayRef<JumpArgState> jumpArgs;
  };
};

class Switch : public InstructionTemplate<Switch, BlockExit, ZeroResult> {
public:
  using Base::Base;
  static void build(IRBuilder &builder, InstructionState &state, Value value,
                    llvm::ArrayRef<Value> caseValues,
                    llvm::ArrayRef<JumpArgState> cases,
                    JumpArgState defaultCase);

  Value getValue() const;
  const Operand &getValueAsOperand() const;
  Value getCaseValue(size_t idx) const;
  const Operand &getCaseValueAsOperand(size_t idx) const;
  JumpArg *getCaseJumpArg(size_t idx) const;
  std::size_t getCaseSize() const;
  JumpArg *getDefaultCase() const;

  static void printer(Switch op, IRPrintContext &context);
  static llvm::StringRef getDebugName() { return "switch"; }

  struct Adaptor {
    Adaptor(llvm::ArrayRef<Value> operands,
            llvm::ArrayRef<JumpArgState> jumpArgs)
        : operands(operands), jumpArgs(jumpArgs) {}

    Value getValue() const;
    Value getCaseValue(size_t idx) const;
    JumpArgState getCaseJumpArg(size_t idx) const;
    JumpArgState getDefaultCase() const;
    std::size_t getCaseSize() const;

    llvm::ArrayRef<Value> operands;
    llvm::ArrayRef<JumpArgState> jumpArgs;
  };
};

class Return : public InstructionTemplate<Return, BlockExit, ZeroResult> {
public:
  using Base::Base;

  static void build(IRBuilder &builder, InstructionState &state);
  static void build(IRBuilder &builder, InstructionState &state, Value value);
  static void build(IRBuilder &builder, InstructionState &state,
                    llvm::ArrayRef<Value> value);

  Value getValue(std::size_t idx) const;
  const Operand &getValueAsOperand(std::size_t idx) const;
  llvm::ArrayRef<Operand> getValues() const;
  std::size_t getValueSize() const { return getValues().size(); }

  static void printer(Return op, IRPrintContext &context);
  static llvm::StringRef getDebugName() { return "return"; }

  struct Adaptor {
    Adaptor(llvm::ArrayRef<Value> operands, llvm::ArrayRef<JumpArgState>)
        : operands(operands) {}

    Value getValue(std::size_t idx) const;
    llvm::ArrayRef<Value> getValues() const;
    std::size_t getValueSize() const { return getValues().size(); }

    llvm::ArrayRef<Value> operands;
  };
};

class Unreachable
    : public InstructionTemplate<Unreachable, BlockExit, ZeroResult> {
public:
  using Base::Base;
  static void build(IRBuilder &builder, InstructionState &state);

  static void printer(Unreachable op, IRPrintContext &context);
  static llvm::StringRef getDebugName() { return "unreachable"; }
  struct Adaptor {
    Adaptor(llvm::ArrayRef<Value>, llvm::ArrayRef<JumpArgState>) {}
  };
};

class Constant : public InstructionTemplate<Constant, Instruction, OneResult> {
public:
  using Base::Base;

  static void build(IRBuilder &builder, InstructionState &state,
                    ConstantAttr value);

  static void printer(Constant op, IRPrintContext &context);
  static llvm::StringRef getDebugName() { return "constant"; }

  ConstantAttr getValue() const;

  void replaceValue(ConstantAttr value);

  struct Adaptor {
    Adaptor(llvm::ArrayRef<Value>, llvm::ArrayRef<JumpArgState>) {}
  };
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
  static llvm::StringRef getDebugName() { return "struct"; }

  struct Adaptor {
    Adaptor(llvm::ArrayRef<Value>, llvm::ArrayRef<JumpArgState>) {}
  };
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
  bool hasInitializer() const { return getInitializer(); }

  static void printer(GlobalVariableDefinition op, IRPrintContext &context);
  static llvm::StringRef getDebugName() { return "global"; }
  struct Adaptor {
    Adaptor(llvm::ArrayRef<Value>, llvm::ArrayRef<JumpArgState>) {}
  };
};

class LocalVariable
    : public InstructionTemplate<LocalVariable, Instruction, OneResult> {
public:
  using Base::Base;
  static void build(IRBuilder &builder, InstructionState &state, Type type);

  static bool verify(LocalVariable op);

  void printAsDef(IRPrintContext &context) const;

  static void printer(LocalVariable op, IRPrintContext &context);
  static llvm::StringRef getDebugName() { return "local"; }

  struct Adaptor {
    Adaptor(llvm::ArrayRef<Value>, llvm::ArrayRef<JumpArgState>) {}
  };
};

class Unresolved
    : public InstructionTemplate<Unresolved, Instruction, OneResult> {
public:
  using Base::Base;
  static void build(IRBuilder &builder, InstructionState &state, Type type);

  static void printer(Unresolved op, IRPrintContext &context);
  static llvm::StringRef getDebugName() { return "unresolved"; }

  struct Adaptor {
    Adaptor(llvm::ArrayRef<Value>, llvm::ArrayRef<JumpArgState>) {}
  };
};

class OutlineConstant
    : public InstructionTemplate<OutlineConstant, Instruction, OneResult> {
public:
  using Base::Base;
  static void build(IRBuilder &builder, InstructionState &state, Value value);

  Value getConstant() const;
  const Operand &getConstantAsOperand() const;

  static void printer(OutlineConstant op, IRPrintContext &context);
  static llvm::StringRef getDebugName() { return "outline_constant"; }

  struct Adaptor {
    Adaptor(llvm::ArrayRef<Value> operands, llvm::ArrayRef<JumpArgState>)
        : operands(operands) {}

    Value getConstant() const;

    llvm::ArrayRef<Value> operands;
  };
};

class InlineCall
    : public InstructionTemplate<InlineCall, Instruction, VariadicResults,
                                 SideEffect, CallLike> {
public:
  using Base::Base;
  static void build(IRBuilder &builder, InstructionState &state,
                    llvm::StringRef name, FunctionT type,
                    llvm::ArrayRef<Value> args);

  llvm::StringRef getName() const;
  FunctionT getFunctionType() const;
  llvm::ArrayRef<Operand> getArguments() const;

  static void printer(InlineCall op, IRPrintContext &context);
  static llvm::StringRef getDebugName() { return "inline_call"; }

  struct Adaptor {
    Adaptor(llvm::ArrayRef<Value> operands, llvm::ArrayRef<JumpArgState>)
        : operands(operands) {}

    llvm::ArrayRef<Value> getArguments() const;

    llvm::ArrayRef<Value> operands;
  };
};

class FunctionArgument
    : public InstructionTemplate<FunctionArgument, Instruction, OneResult> {
public:
  using Base::Base;

  static void build(IRBuilder &builder, InstructionState &state, Type type);

  static void printer(FunctionArgument op, IRPrintContext &context);
  static llvm::StringRef getDebugName() { return "arg"; }

  struct Adaptor {
    Adaptor(llvm::ArrayRef<Value>, llvm::ArrayRef<JumpArgState>) {}
  };
};

class MemCpy : public InstructionTemplate<MemCpy, Instruction, ZeroResult,
                                          SideEffect, CallLike> {
public:
  using Base::Base;
  static void build(IRBuilder &builder, InstructionState &state, Value dest,
                    Value src, Value size);

  Value getDest() const;
  const Operand &getDestAsOperand() const;
  Value getSrc() const;
  const Operand &getSrcAsOperand() const;
  Value getSize() const;
  const Operand &getSizeAsOperand() const;
  static void printer(MemCpy op, IRPrintContext &context);
  static llvm::StringRef getDebugName() { return "memcpy"; }

  struct Adaptor {
    Adaptor(llvm::ArrayRef<Value> operands, llvm::ArrayRef<JumpArgState>)
        : operands(operands) {}

    Value getDest() const;
    Value getSrc() const;
    Value getSize() const;

    llvm::ArrayRef<Value> operands;
  };
};

void registerBuiltinInstructions(IRContext *context);

} // namespace kecc::ir::inst

#endif // KECC_IR_IR_INSTRUCTIONS_H
