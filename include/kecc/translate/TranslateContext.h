#ifndef KECC_TRANSLATE_CONTEXT_H
#define KECC_TRANSLATE_CONTEXT_H

#include "kecc/asm/Register.h"
#include "kecc/utils/LogicalResult.h"

namespace kecc {

class TranslateRuleSet;

struct TranslateContextImpl;
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

  as::AnonymousRegisterStorage *getAnonymousRegStorage();

  TranslateRuleSet *getTranslateRuleSet();

private:
  std::unique_ptr<TranslateContextImpl> impl;
};

} // namespace kecc

#endif // KECC_TRANSLATE_CONTEXT_H
