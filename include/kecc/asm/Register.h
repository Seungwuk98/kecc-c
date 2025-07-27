#ifndef KECC_ASM_REGISTER_H
#define KECC_ASM_REGISTER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace kecc::as {

class RegisterImpl;
class Register {
public:
  enum class Type { Integer, FloatingPoint };

  std::string toString() const;

  static Register zero();
  static Register ra();
  static Register sp();
  static Register gp();
  static Register tp();
  static Register t0();
  static Register t1();
  static Register t2();
  static Register t3();
  static Register t4();
  static Register t5();
  static Register t6();

  static Register s0();
  static Register s1();
  static Register s2();
  static Register s3();
  static Register s4();
  static Register s5();
  static Register s6();
  static Register s7();
  static Register s8();
  static Register s9();
  static Register s10();
  static Register s11();

  static Register a0();
  static Register a1();
  static Register a2();
  static Register a3();
  static Register a4();
  static Register a5();
  static Register a6();
  static Register a7();

  static Register ft0();
  static Register ft1();
  static Register ft2();
  static Register ft3();
  static Register ft4();
  static Register ft5();
  static Register ft6();
  static Register ft7();
  static Register ft8();
  static Register ft9();
  static Register ft10();
  static Register ft11();

  static Register fs0();
  static Register fs1();
  static Register fs2();
  static Register fs3();
  static Register fs4();
  static Register fs5();
  static Register fs6();
  static Register fs7();
  static Register fs8();
  static Register fs9();
  static Register fs10();
  static Register fs11();

  static Register fa0();
  static Register fa1();
  static Register fa2();
  static Register fa3();
  static Register fa4();
  static Register fa5();
  static Register fa6();
  static Register fa7();

  static Register temp(Type type, size_t index);
  static Register arg(Type type, size_t index);
  static Register saved(Type type, size_t index);

  std::optional<std::pair<Type, size_t>> getTemp() const;
  std::optional<std::pair<Type, size_t>> getArg() const;
  std::optional<std::pair<Type, size_t>> getSaved() const;

  bool operator==(const Register &other) const { return impl == other.impl; }
  bool operator!=(const Register &other) const { return impl != other.impl; }

  bool isCallerSaved() const;
  bool isCalleeSaved() const;
  llvm::StringRef getDescription() const;
  bool isXRegister() const;
  bool isFRegister() const;

  bool isInteger() const;
  bool isFloatingPoint() const;

  friend inline llvm::hash_code hash_value(const Register &reg) {
    return llvm::DenseMapInfo<RegisterImpl *>::getHashValue(reg.impl);
  }

private:
  friend class llvm::DenseMapInfo<Register>;
  Register(RegisterImpl *impl) : impl(impl) {}

  RegisterImpl *impl;
};

llvm::ArrayRef<Register> getIntTempRegisters();
llvm::ArrayRef<Register> getIntArgRegisters();
llvm::ArrayRef<Register> getIntSavedRegisters();
llvm::ArrayRef<Register> getFpTempRegisters();
llvm::ArrayRef<Register> getFpArgRegisters();
llvm::ArrayRef<Register> getFpSavedRegisters();

} // namespace kecc::as

namespace llvm {

template <> struct DenseMapInfo<kecc::as::Register> {
  static kecc::as::Register getEmptyKey() {
    return llvm::DenseMapInfo<kecc::as::RegisterImpl *>::getEmptyKey();
  }

  static kecc::as::Register getTombstoneKey() {
    return llvm::DenseMapInfo<kecc::as::RegisterImpl *>::getTombstoneKey();
  }

  static unsigned getHashValue(const kecc::as::Register &reg) {
    return hash_value(reg);
  }

  static bool isEqual(const kecc::as::Register &l,
                      const kecc::as::Register &r) {
    return l == r;
  }
};

} // namespace llvm

#endif // KECC_ASM_REGISTER_H
