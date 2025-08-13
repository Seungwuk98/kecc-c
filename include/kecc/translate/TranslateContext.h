#ifndef KECC_TRANSLATE_CONTEXT_H
#define KECC_TRANSLATE_CONTEXT_H

#include "kecc/asm/Register.h"
#include "kecc/utils/LogicalResult.h"

namespace kecc {

class TranslateContext {
public:
  TranslateContext() = default;

  llvm::ArrayRef<as::Register> getTempRegisters() const;

  llvm::ArrayRef<as::Register>
  getRegistersForAllocate(as::RegisterType regType);

  void setTempRegisters(llvm::ArrayRef<as::Register> regs);
  void setRegistersForAllocate(llvm::ArrayRef<as::Register> intRegs,
                               llvm::ArrayRef<as::Register> floatRegs);

  utils::LogicalResult setRegistersFromOption(llvm::StringRef option);

  as::AnonymousRegisterStorage *getAnonymousRegStorage() {
    return &anonymousRegStorage;
  }

private:
  llvm::SmallVector<as::Register, 32> tempIntRegisters;
  llvm::SmallVector<as::Register, 32> registersForAllocateInt;
  llvm::SmallVector<as::Register, 32> registersForAllocateFloat;

  as::AnonymousRegisterStorage anonymousRegStorage;
};

} // namespace kecc

#endif // KECC_TRANSLATE_CONTEXT_H
