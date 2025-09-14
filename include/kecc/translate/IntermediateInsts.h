#ifndef KECC_TRANSLATE_INTERMEDIATE_INSTS_H
#define KECC_TRANSLATE_INTERMEDIATE_INSTS_H

#include "kecc/ir/IRBuilder.h"
#include "kecc/ir/Instruction.h"
#include "kecc/ir/Type.h"
#include "llvm/ADT/ArrayRef.h"

namespace kecc {
namespace translate::inst {

class LoadOffset
    : public ir::InstructionTemplate<LoadOffset, ir::Instruction, ir::OneResult,
                                     ir::ReadMemory> {
public:
  using Base::Base;

  static void build(ir::IRBuilder &builder, ir::InstructionState &state,
                    ir::Value ptr, std::int64_t offset, ir::Type loadType);

  ir::Value getPointer() const;
  const ir::Operand &getPointerAsOperand() const;
  std::int64_t getOffset() const;

  static void printer(LoadOffset op, ir::IRPrintContext &context);

  struct Adaptor {
    Adaptor(llvm::ArrayRef<ir::Value> operands,
            llvm::ArrayRef<ir::JumpArgState>)
        : operands(operands) {}

    ir::Value getPointer() const { return operands[0]; }

    llvm::ArrayRef<ir::Value> operands;
  };
};

class StoreOffset
    : public ir::InstructionTemplate<StoreOffset, ir::Instruction,
                                     ir::ZeroResult, ir::WriteMemory,
                                     ir::SideEffect> {
public:
  using Base::Base;

  static void build(ir::IRBuilder &builder, ir::InstructionState &state,
                    ir::Value value, ir::Value base, std::int64_t offset);

  ir::Value getValue() const { return getValueAsOperand(); }
  ir::Value getPointer() const { return getPointerAsOperand(); }
  const ir::Operand &getValueAsOperand() const;
  const ir::Operand &getPointerAsOperand() const;
  std::int64_t getOffset() const;

  static void printer(StoreOffset op, ir::IRPrintContext &context);

  struct Adaptor {
    Adaptor(llvm::ArrayRef<ir::Value> operands,
            llvm::ArrayRef<ir::JumpArgState>)
        : operands(operands) {}

    ir::Value getValue() const { return operands[0]; }
    ir::Value getPointer() const { return operands[1]; }

    llvm::ArrayRef<ir::Value> operands;
  };
};

class Copy : public ir::InstructionTemplate<Copy, ir::Instruction,
                                            ir::OneResult, ir::Pure> {
public:
  using Base::Base;
  static void build(ir::IRBuilder &builder, ir::InstructionState &state,
                    ir::Value value, ir::Type type);

  ir::Value getValue() const;
  const ir::Operand &getValueAsOperand() const;

  static void printer(Copy op, ir::IRPrintContext &context);

  struct Adaptor {
    Adaptor(llvm::ArrayRef<ir::Value> operands,
            llvm::ArrayRef<ir::JumpArgState>)
        : operands(operands) {}

    ir::Value getValue() const;

    llvm::ArrayRef<ir::Value> operands;
  };
};

} // namespace translate::inst
} // namespace kecc

DECLARE_KECC_TYPE_ID(kecc::translate::inst::LoadOffset)
DECLARE_KECC_TYPE_ID(kecc::translate::inst::StoreOffset)
DECLARE_KECC_TYPE_ID(kecc::translate::inst::Copy)

#endif // KECC_TRANSLATE_INTERMEDIATE_INSTS_H
