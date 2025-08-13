#include "kecc/asm/Register.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/ManagedStatic.h"

namespace kecc::as {

class RegisterImpl {
public:
  RegisterImpl(Mnemonic mnemonic, int index, RegisterType type, ABIKind kind,
               CallingConvension callingConvention, int abiIndex,
               llvm::StringRef description, llvm::StringRef printName)
      : mnemonic(mnemonic), type(type), kind(kind),
        callingConvention(callingConvention), index(index),
        description(description), printName(printName) {}

  Mnemonic getMnemonic() const { return mnemonic; }
  int getIndex() const { return index; }
  RegisterType getType() const { return type; }
  ABIKind getABIKind() const { return kind; }
  CallingConvension getCallingConvention() const { return callingConvention; }
  int getABIIndex() const { return index; }
  llvm::StringRef getDescription() const { return description; }
  llvm::StringRef getPrintName() const { return printName; }

private:
  Mnemonic mnemonic;
  int index;
  RegisterType type;
  ABIKind kind;
  CallingConvension callingConvention;

  std::string description;
  std::string printName;
};

namespace detail {

//===----------------------------------------------------------------------===//
/// x
//===----------------------------------------------------------------------===//

static RegisterImpl x0(Mnemonic::x, 0, RegisterType::Integer, ABIKind::Zero,
                       CallingConvension::None, -1, "Hard-wired zero", "zero");
static RegisterImpl x1(Mnemonic::x, 1, RegisterType::Integer, ABIKind::Ra,
                       CallingConvension::CallerSave, -1, "Return address",
                       "ra");
static RegisterImpl x2(Mnemonic::x, 2, RegisterType::Integer, ABIKind::Sp,
                       CallingConvension::CalleeSave, -1, "Stack pointer",
                       "sp");
static RegisterImpl x3(Mnemonic::x, 3, RegisterType::Integer, ABIKind::Gp,
                       CallingConvension::None, -1, "Global pointer", "gp");
static RegisterImpl x4(Mnemonic::x, 4, RegisterType::Integer, ABIKind::Tp,
                       CallingConvension::None, -1, "Thread pointer", "tp");
static RegisterImpl x5(Mnemonic::x, 5, RegisterType::Integer, ABIKind::Temp,
                       CallingConvension::CallerSave, 0,
                       "Temporary/alternate link register", "t0");
static RegisterImpl x6(Mnemonic::x, 6, RegisterType::Integer, ABIKind::Temp,
                       CallingConvension::CallerSave, 1, "Temporary register",
                       "t1");
static RegisterImpl x7(Mnemonic::x, 7, RegisterType::Integer, ABIKind::Temp,
                       CallingConvension::CallerSave, 2, "Temporary register",
                       "t2");
static RegisterImpl x8(Mnemonic::x, 8, RegisterType::Integer, ABIKind::Temp,
                       CallingConvension::CalleeSave, 0,
                       "Saved register/frame pointer", "s0");
static RegisterImpl x9(Mnemonic::x, 9, RegisterType::Integer, ABIKind::Temp,
                       CallingConvension::CalleeSave, 1, "Saved register",
                       "s1");
static RegisterImpl x10(Mnemonic::x, 10, RegisterType::Integer, ABIKind::Temp,
                        CallingConvension::CallerSave, 0,
                        "Function arguments/return values", "a0");
static RegisterImpl x11(Mnemonic::x, 11, RegisterType::Integer, ABIKind::Temp,
                        CallingConvension::CallerSave, 1,
                        "Function arguments/return values", "a1");
static RegisterImpl x12(Mnemonic::x, 12, RegisterType::Integer, ABIKind::Temp,
                        CallingConvension::CallerSave, 2, "Function arguments",
                        "a2");
static RegisterImpl x13(Mnemonic::x, 13, RegisterType::Integer, ABIKind::Temp,
                        CallingConvension::CallerSave, 3, "Function arguments",
                        "a3");
static RegisterImpl x14(Mnemonic::x, 14, RegisterType::Integer, ABIKind::Temp,
                        CallingConvension::CallerSave, 4, "Function arguments",
                        "a4");
static RegisterImpl x15(Mnemonic::x, 15, RegisterType::Integer, ABIKind::Temp,
                        CallingConvension::CallerSave, 5, "Function arguments",
                        "a5");
static RegisterImpl x16(Mnemonic::x, 16, RegisterType::Integer, ABIKind::Temp,
                        CallingConvension::CallerSave, 6, "Function arguments",
                        "a6");
static RegisterImpl x17(Mnemonic::x, 17, RegisterType::Integer, ABIKind::Temp,
                        CallingConvension::CallerSave, 7, "Function arguments",
                        "a7");
static RegisterImpl x18(Mnemonic::x, 18, RegisterType::Integer, ABIKind::Saved,
                        CallingConvension::CalleeSave, 2, "Saved register",
                        "s2");
static RegisterImpl x19(Mnemonic::x, 19, RegisterType::Integer, ABIKind::Saved,
                        CallingConvension::CalleeSave, 3, "Saved register",
                        "s3");
static RegisterImpl x20(Mnemonic::x, 20, RegisterType::Integer, ABIKind::Saved,
                        CallingConvension::CalleeSave, 4, "Saved register",
                        "s4");
static RegisterImpl x21(Mnemonic::x, 21, RegisterType::Integer, ABIKind::Saved,
                        CallingConvension::CalleeSave, 5, "Saved register",
                        "s5");
static RegisterImpl x22(Mnemonic::x, 22, RegisterType::Integer, ABIKind::Saved,
                        CallingConvension::CalleeSave, 6, "Saved register",
                        "s6");
static RegisterImpl x23(Mnemonic::x, 23, RegisterType::Integer, ABIKind::Saved,
                        CallingConvension::CalleeSave, 7, "Saved register",
                        "s7");
static RegisterImpl x24(Mnemonic::x, 24, RegisterType::Integer, ABIKind::Saved,
                        CallingConvension::CalleeSave, 8, "Saved register",
                        "s8");
static RegisterImpl x25(Mnemonic::x, 25, RegisterType::Integer, ABIKind::Saved,
                        CallingConvension::CalleeSave, 9, "Saved register",
                        "s9");
static RegisterImpl x26(Mnemonic::x, 26, RegisterType::Integer, ABIKind::Saved,
                        CallingConvension::CalleeSave, 10, "Saved register",
                        "s10");
static RegisterImpl x27(Mnemonic::x, 27, RegisterType::Integer, ABIKind::Saved,
                        CallingConvension::CalleeSave, 11, "Saved register",
                        "s11");
static RegisterImpl x28(Mnemonic::x, 28, RegisterType::Integer, ABIKind::Temp,
                        CallingConvension::CallerSave, 3, "Temporary register",
                        "t3");
static RegisterImpl x29(Mnemonic::x, 29, RegisterType::Integer, ABIKind::Temp,
                        CallingConvension::CallerSave, 4, "Temporary register",
                        "t4");
static RegisterImpl x30(Mnemonic::x, 30, RegisterType::Integer, ABIKind::Temp,
                        CallingConvension::CallerSave, 5, "Temporary register",
                        "t5");
static RegisterImpl x31(Mnemonic::x, 31, RegisterType::Integer, ABIKind::Temp,
                        CallingConvension::CallerSave, 6, "Temporary register",
                        "t6");

//===----------------------------------------------------------------------===//
/// f
//===----------------------------------------------------------------------===//

static RegisterImpl f0(Mnemonic::f, 0, RegisterType::FloatingPoint,
                       ABIKind::Temp, CallingConvension::CallerSave, 0,
                       "FP temporary", "ft0");
static RegisterImpl f1(Mnemonic::f, 1, RegisterType::FloatingPoint,
                       ABIKind::Temp, CallingConvension::CallerSave, 1,
                       "FP temporary", "ft1");
static RegisterImpl f2(Mnemonic::f, 2, RegisterType::FloatingPoint,
                       ABIKind::Temp, CallingConvension::CallerSave, 2,
                       "FP temporary", "ft2");
static RegisterImpl f3(Mnemonic::f, 3, RegisterType::FloatingPoint,
                       ABIKind::Temp, CallingConvension::CallerSave, 3,
                       "FP temporary", "ft3");
static RegisterImpl f4(Mnemonic::f, 4, RegisterType::FloatingPoint,
                       ABIKind::Temp, CallingConvension::CallerSave, 4,
                       "FP temporary", "ft4");
static RegisterImpl f5(Mnemonic::f, 5, RegisterType::FloatingPoint,
                       ABIKind::Temp, CallingConvension::CallerSave, 5,
                       "FP temporary", "ft5");
static RegisterImpl f6(Mnemonic::f, 6, RegisterType::FloatingPoint,
                       ABIKind::Temp, CallingConvension::CallerSave, 6,
                       "FP temporary", "ft6");
static RegisterImpl f7(Mnemonic::f, 7, RegisterType::FloatingPoint,
                       ABIKind::Temp, CallingConvension::CallerSave, 7,
                       "FP temporary", "ft7");
static RegisterImpl f8(Mnemonic::f, 8, RegisterType::FloatingPoint,
                       ABIKind::Saved, CallingConvension::CalleeSave, 0,
                       "FP saved register", "fs0");
static RegisterImpl f9(Mnemonic::f, 9, RegisterType::FloatingPoint,
                       ABIKind::Saved, CallingConvension::CalleeSave, 1,
                       "FP saved register", "fs1");
static RegisterImpl f10(Mnemonic::f, 10, RegisterType::FloatingPoint,
                        ABIKind::Arg, CallingConvension::CallerSave, 0,
                        "FP argument/return value", "fa0");
static RegisterImpl f11(Mnemonic::f, 11, RegisterType::FloatingPoint,
                        ABIKind::Arg, CallingConvension::CallerSave, 1,
                        "FP argument/return value", "fa1");
static RegisterImpl f12(Mnemonic::f, 12, RegisterType::FloatingPoint,
                        ABIKind::Arg, CallingConvension::CallerSave, 2,
                        "FP argument", "fa2");
static RegisterImpl f13(Mnemonic::f, 13, RegisterType::FloatingPoint,
                        ABIKind::Arg, CallingConvension::CallerSave, 3,
                        "FP argument", "fa3");
static RegisterImpl f14(Mnemonic::f, 14, RegisterType::FloatingPoint,
                        ABIKind::Arg, CallingConvension::CallerSave, 4,
                        "FP argument", "fa4");
static RegisterImpl f15(Mnemonic::f, 15, RegisterType::FloatingPoint,
                        ABIKind::Arg, CallingConvension::CallerSave, 5,
                        "FP argument", "fa5");
static RegisterImpl f16(Mnemonic::f, 16, RegisterType::FloatingPoint,
                        ABIKind::Arg, CallingConvension::CallerSave, 6,
                        "FP argument", "fa6");
static RegisterImpl f17(Mnemonic::f, 17, RegisterType::FloatingPoint,
                        ABIKind::Arg, CallingConvension::CallerSave, 7,
                        "FP argument", "fa7");
static RegisterImpl f18(Mnemonic::f, 18, RegisterType::FloatingPoint,
                        ABIKind::Saved, CallingConvension::CalleeSave, 2,
                        "FP saved register", "fs2");
static RegisterImpl f19(Mnemonic::f, 19, RegisterType::FloatingPoint,
                        ABIKind::Saved, CallingConvension::CalleeSave, 3,
                        "FP saved register", "fs3");
static RegisterImpl f20(Mnemonic::f, 20, RegisterType::FloatingPoint,
                        ABIKind::Saved, CallingConvension::CalleeSave, 4,
                        "FP saved register", "fs4");
static RegisterImpl f21(Mnemonic::f, 21, RegisterType::FloatingPoint,
                        ABIKind::Saved, CallingConvension::CalleeSave, 5,
                        "FP saved register", "fs5");
static RegisterImpl f22(Mnemonic::f, 22, RegisterType::FloatingPoint,
                        ABIKind::Saved, CallingConvension::CalleeSave, 6,
                        "FP saved register", "fs6");
static RegisterImpl f23(Mnemonic::f, 23, RegisterType::FloatingPoint,
                        ABIKind::Saved, CallingConvension::CalleeSave, 7,
                        "FP saved register", "fs7");
static RegisterImpl f24(Mnemonic::f, 24, RegisterType::FloatingPoint,
                        ABIKind::Saved, CallingConvension::CalleeSave, 8,
                        "FP saved register", "fs8");
static RegisterImpl f25(Mnemonic::f, 25, RegisterType::FloatingPoint,
                        ABIKind::Saved, CallingConvension::CalleeSave, 9,
                        "FP saved register", "fs9");
static RegisterImpl f26(Mnemonic::f, 26, RegisterType::FloatingPoint,
                        ABIKind::Saved, CallingConvension::CalleeSave, 10,
                        "FP saved register", "fs10");
static RegisterImpl f27(Mnemonic::f, 27, RegisterType::FloatingPoint,
                        ABIKind::Saved, CallingConvension::CalleeSave, 11,
                        "FP saved register", "fs11");
static RegisterImpl f28(Mnemonic::f, 28, RegisterType::FloatingPoint,
                        ABIKind::Temp, CallingConvension::CallerSave, 8,
                        "FP temporary", "ft8");
static RegisterImpl f29(Mnemonic::f, 29, RegisterType::FloatingPoint,
                        ABIKind::Temp, CallingConvension::CallerSave, 9,
                        "FP temporary", "ft9");
static RegisterImpl f30(Mnemonic::f, 30, RegisterType::FloatingPoint,
                        ABIKind::Temp, CallingConvension::CallerSave, 10,
                        "FP temporary", "ft10");
static RegisterImpl f31(Mnemonic::f, 31, RegisterType::FloatingPoint,
                        ABIKind::Temp, CallingConvension::CallerSave, 11,
                        "FP temporary", "ft11");
} // namespace detail

Register Register::zero() { return &detail::x0; }
Register Register::ra() { return &detail::x1; }
Register Register::sp() { return &detail::x2; }
Register Register::gp() { return &detail::x3; }
Register Register::tp() { return &detail::x4; }
Register Register::t0() { return &detail::x5; }
Register Register::t1() { return &detail::x6; }
Register Register::t2() { return &detail::x7; }
Register Register::t3() { return &detail::x28; }
Register Register::t4() { return &detail::x29; }
Register Register::t5() { return &detail::x30; }
Register Register::t6() { return &detail::x31; }
Register Register::s0() { return &detail::x8; }
Register Register::s1() { return &detail::x9; }
Register Register::s2() { return &detail::x18; }
Register Register::s3() { return &detail::x19; }
Register Register::s4() { return &detail::x20; }
Register Register::s5() { return &detail::x21; }
Register Register::s6() { return &detail::x22; }
Register Register::s7() { return &detail::x23; }
Register Register::s8() { return &detail::x24; }
Register Register::s9() { return &detail::x25; }
Register Register::s10() { return &detail::x26; }
Register Register::s11() { return &detail::x27; }
Register Register::a0() { return &detail::x10; }
Register Register::a1() { return &detail::x11; }
Register Register::a2() { return &detail::x12; }
Register Register::a3() { return &detail::x13; }
Register Register::a4() { return &detail::x14; }
Register Register::a5() { return &detail::x15; }
Register Register::a6() { return &detail::x16; }
Register Register::a7() { return &detail::x17; }

Register Register::ft0() { return &detail::f0; }
Register Register::ft1() { return &detail::f1; }
Register Register::ft2() { return &detail::f2; }
Register Register::ft3() { return &detail::f3; }
Register Register::ft4() { return &detail::f4; }
Register Register::ft5() { return &detail::f5; }
Register Register::ft6() { return &detail::f6; }
Register Register::ft7() { return &detail::f7; }
Register Register::ft8() { return &detail::f28; }
Register Register::ft9() { return &detail::f29; }
Register Register::ft10() { return &detail::f30; }
Register Register::ft11() { return &detail::f31; }
Register Register::fs0() { return &detail::f8; }
Register Register::fs1() { return &detail::f9; }
Register Register::fs2() { return &detail::f18; }
Register Register::fs3() { return &detail::f19; }
Register Register::fs4() { return &detail::f20; }
Register Register::fs5() { return &detail::f21; }
Register Register::fs6() { return &detail::f22; }
Register Register::fs7() { return &detail::f23; }
Register Register::fs8() { return &detail::f24; }
Register Register::fs9() { return &detail::f25; }
Register Register::fs10() { return &detail::f26; }
Register Register::fs11() { return &detail::f27; }
Register Register::fa0() { return &detail::f10; }
Register Register::fa1() { return &detail::f11; }
Register Register::fa2() { return &detail::f12; }
Register Register::fa3() { return &detail::f13; }
Register Register::fa4() { return &detail::f14; }
Register Register::fa5() { return &detail::f15; }
Register Register::fa6() { return &detail::f16; }
Register Register::fa7() { return &detail::f17; }

namespace detail {

llvm::ArrayRef<RegisterImpl *> xRegisters = {
    &x0,  &x1,  &x2,  &x3,  &x4,  &x5,  &x6,  &x7,  &x8,  &x9,  &x10,
    &x11, &x12, &x13, &x14, &x15, &x16, &x17, &x18, &x19, &x20, &x21,
    &x22, &x23, &x24, &x25, &x26, &x27, &x28, &x29, &x30, &x31,
};

llvm::ArrayRef<RegisterImpl *> fRegisters = {
    &f0,  &f1,  &f2,  &f3,  &f4,  &f5,  &f6,  &f7,  &f8,  &f9,  &f10,
    &f11, &f12, &f13, &f14, &f15, &f16, &f17, &f18, &f19, &f20, &f21,
    &f22, &f23, &f24, &f25, &f26, &f27, &f28, &f29, &f30, &f31,
};

llvm::ArrayRef<Register> integerTempRegisters = {
    Register::t0(), Register::t1(), Register::t2(), Register::t3(),
    Register::t4(), Register::t5(), Register::t6(),
};

llvm::ArrayRef<Register> integerSavedRegisters = {
    Register::s0(), Register::s1(), Register::s2(),  Register::s3(),
    Register::s4(), Register::s5(), Register::s6(),  Register::s7(),
    Register::s8(), Register::s9(), Register::s10(), Register::s11(),
};

llvm::ArrayRef<Register> integerArgRegisters = {
    Register::a0(), Register::a1(), Register::a2(), Register::a3(),
    Register::a4(), Register::a5(), Register::a6(), Register::a7(),
};

llvm::ArrayRef<Register> floatTempRegisters = {
    Register::ft0(), Register::ft1(), Register::ft2(),  Register::ft3(),
    Register::ft4(), Register::ft5(), Register::ft6(),  Register::ft7(),
    Register::ft8(), Register::ft9(), Register::ft10(), Register::ft11(),
};

llvm::ArrayRef<Register> floatSavedRegisters = {
    Register::fs0(), Register::fs1(), Register::fs2(),  Register::fs3(),
    Register::fs4(), Register::fs5(), Register::fs6(),  Register::fs7(),
    Register::fs8(), Register::fs9(), Register::fs10(), Register::fs11(),
};

llvm::ArrayRef<Register> floatArgRegisters = {
    Register::fa0(), Register::fa1(), Register::fa2(), Register::fa3(),
    Register::fa4(), Register::fa5(), Register::fa6(), Register::fa7(),
};

} // namespace detail

std::string Register::toString() const { return impl->getPrintName().str(); }
bool Register::isCalleeSaved() const {
  return impl->getCallingConvention() == CallingConvension::CalleeSave;
}
bool Register::isCallerSaved() const {
  return impl->getCallingConvention() == CallingConvension::CallerSave;
}
llvm::StringRef Register::getDescription() const {
  return impl->getDescription();
}
bool Register::isXRegister() const {
  return impl->getMnemonic() == Mnemonic::x;
}
bool Register::isFRegister() const {
  return impl->getMnemonic() == Mnemonic::f;
}
bool Register::isInteger() const {
  return impl->getType() == RegisterType::Integer;
}
bool Register::isFloatingPoint() const {
  return impl->getType() == RegisterType::FloatingPoint;
}

std::optional<std::pair<RegisterType, std::size_t>> Register::getTemp() const {
  if (impl->getABIKind() != ABIKind::Temp)
    return std::nullopt;

  return std::pair(impl->getType(), impl->getABIIndex());
}

std::optional<std::pair<RegisterType, std::size_t>> Register::getSaved() const {
  if (impl->getABIKind() != ABIKind::Saved)
    return std::nullopt;

  return std::pair(impl->getType(), impl->getABIIndex());
}

std::optional<std::pair<RegisterType, std::size_t>> Register::getArg() const {
  if (impl->getABIKind() != ABIKind::Arg)
    return std::nullopt;

  return std::pair(impl->getType(), impl->getABIIndex());
}

Register Register::temp(RegisterType type, size_t index) {
  if (type == RegisterType::Integer) {
    assert(index < detail::integerTempRegisters.size() &&
           "Index out of bounds for integer temporary registers");
    return detail::integerTempRegisters[index];
  } else {
    assert(index < detail::floatTempRegisters.size() &&
           "Index out of bounds for floating point temporary registers");
    return detail::floatTempRegisters[index];
  }
}

Register Register::saved(RegisterType type, size_t index) {
  if (type == RegisterType::Integer) {
    assert(index < detail::integerSavedRegisters.size() &&
           "Index out of bounds for integer saved registers");
    return detail::integerSavedRegisters[index];
  } else {
    assert(index < detail::floatSavedRegisters.size() &&
           "Index out of bounds for floating point saved registers");
    return detail::floatSavedRegisters[index];
  }
}

Register Register::arg(RegisterType type, size_t index) {
  if (type == RegisterType::Integer) {
    assert(index < detail::integerArgRegisters.size() &&
           "Index out of bounds for integer argument registers");
    return detail::integerArgRegisters[index];
  } else {
    assert(index < detail::floatArgRegisters.size() &&
           "Index out of bounds for floating point argument registers");
    return detail::floatArgRegisters[index];
  }
}

llvm::ArrayRef<Register> getIntTempRegisters() {
  return detail::integerTempRegisters;
}

llvm::ArrayRef<Register> getIntArgRegisters() {
  return detail::integerArgRegisters;
}

llvm::ArrayRef<Register> getIntSavedRegisters() {
  return detail::integerSavedRegisters;
}

llvm::ArrayRef<Register> getFpTempRegisters() {
  return detail::floatTempRegisters;
}

llvm::ArrayRef<Register> getFpArgRegisters() {
  return detail::floatArgRegisters;
}

llvm::ArrayRef<Register> getFpSavedRegisters() {
  return detail::floatSavedRegisters;
}

bool Register::definedAnonymousRegister(AnonymousRegisterStorage *storage,
                                        llvm::StringRef name) {
  return storage->contains(name);
}

Register Register::createAnonymousRegister(AnonymousRegisterStorage *storage,
                                           RegisterType type,
                                           CallingConvension callingConvention,
                                           llvm::StringRef name) {
  assert(!name.empty() > 0 && "Anonymous register must have a non-empty name");
  assert(!storage->contains(name) &&
         "Register with this name already exists in the storage");

  RegisterImpl *impl = new RegisterImpl(static_cast<Mnemonic>(-1), -1, type,
                                        static_cast<ABIKind>(-1),
                                        callingConvention, -1, "", name);

  auto reg = storage->insert(std::unique_ptr<RegisterImpl>(impl));
  return reg;
}

Register Register::getAnonymousRegister(AnonymousRegisterStorage *storage,
                                        llvm::StringRef name) {
  assert(!name.empty() > 0 && "Anonymous register must have a non-empty name");
  return storage->get(name);
}

Register Register::getX(size_t index) { return detail::xRegisters[index]; }
Register Register::getF(size_t index) { return detail::fRegisters[index]; }

AnonymousRegisterStorage::AnonymousRegisterStorage() {}
AnonymousRegisterStorage::~AnonymousRegisterStorage() = default;

bool AnonymousRegisterStorage::contains(llvm::StringRef name) {
  return impls.contains(name);
}

Register AnonymousRegisterStorage::insert(std::unique_ptr<RegisterImpl> impl) {
  llvm::StringRef printName = impl->getPrintName();
  auto [it, inserted] = impls.try_emplace(printName, std::move(impl));
  assert(inserted);
  return it->second.get();
}

Register AnonymousRegisterStorage::get(llvm::StringRef name) {
  return impls.at(name).get();
}

} // namespace kecc::as
