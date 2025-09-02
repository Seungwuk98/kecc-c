#include "kecc/translate/TranslateContext.h"
#include "kecc/asm/Register.h"
#include "kecc/asm/RegisterParser.h"
#include "kecc/translate/IRTranslater.h"
#include "kecc/utils/LogicalResult.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include <format>

namespace kecc {

struct TranslateContextImpl {
  TranslationRuleSet translateRules;

  llvm::SmallVector<as::Register, 32> tempIntRegisters;
  llvm::SmallVector<as::Register, 32> registersForAllocateInt;
  llvm::SmallVector<as::Register, 32> registersForAllocateFloat;

  as::AnonymousRegisterStorage anonymousRegStorage;
};

TranslateContext::TranslateContext()
    : impl(std::make_unique<TranslateContextImpl>()) {}

TranslateContext::~TranslateContext() = default;

llvm::ArrayRef<as::Register> TranslateContext::getTempRegisters() const {
  return impl->tempIntRegisters;
}

llvm::ArrayRef<as::Register>
TranslateContext::getRegistersForAllocate(as::RegisterType regType) {
  return (regType == as::RegisterType::Integer)
             ? impl->registersForAllocateInt
             : impl->registersForAllocateFloat;
}

void TranslateContext::setTempRegisters(llvm::ArrayRef<as::Register> regs) {
  assert(llvm::all_of(
             regs, [&](const as::Register &reg) { return reg.isInteger(); }) &&
         "Temporary registers must be integer registers");

  impl->tempIntRegisters.clear();
  impl->tempIntRegisters.append(regs.begin(), regs.end());
}

void TranslateContext::setRegistersForAllocate(
    llvm::ArrayRef<as::Register> intRegs,
    llvm::ArrayRef<as::Register> floatRegs) {
  impl->registersForAllocateInt.clear();
  impl->registersForAllocateInt.append(intRegs.begin(), intRegs.end());
  impl->registersForAllocateFloat.clear();
  impl->registersForAllocateFloat.append(floatRegs.begin(), floatRegs.end());
}

static llvm::SmallVector<as::Register>
findDuplicatedRegisters(llvm::ArrayRef<as::Register> registers) {
  llvm::DenseSet<as::Register> seen;
  llvm::SmallVector<as::Register> duplicates;

  for (const auto &reg : registers) {
    if (!seen.insert(reg).second)
      duplicates.emplace_back(reg);
  }

  return duplicates;
}

std::string joinRegs(llvm::ArrayRef<as::Register> reg) {
  std::string result;
  llvm::raw_string_ostream ss(result);
  for (const auto &r : reg) {
    if (!result.empty())
      ss << ", ";
    ss << r;
  }
  return result;
}

utils::LogicalResult
TranslateContext::setRegistersFromOption(llvm::StringRef option) {
  llvm::SourceMgr srcMgr;

  auto memBuffer =
      llvm::MemoryBuffer::getMemBuffer(option, "<register option>");

  auto index = srcMgr.AddNewSourceBuffer(std::move(memBuffer), llvm::SMLoc());
  llvm::StringRef content = srcMgr.getMemoryBuffer(index)->getBuffer();
  as::RegisterParser parser(&impl->anonymousRegStorage, srcMgr, content);

  auto registerOptions = parser.parseRegiserOptions();
  if (!registerOptions)
    return utils::LogicalResult::failure();

  if (!registerOptions->contains("temp")) {
    srcMgr.PrintMessage(
        llvm::errs(), llvm::SMLoc::getFromPointer(content.data()),
        llvm::SourceMgr::DK_Error, "Missing 'temp' register option");
    return utils::LogicalResult::failure();
  }

  auto tempRegisters = registerOptions->find("temp")->second;

  if (llvm::any_of(tempRegisters, [](const as::Register &reg) {
        return reg.isFloatingPoint();
      })) {
    srcMgr.PrintMessage(llvm::errs(),
                        llvm::SMLoc::getFromPointer(content.data()),
                        llvm::SourceMgr::DK_Error,
                        "Temporary registers must be integer registers");
    return utils::LogicalResult::failure();
  }

  if (auto duplicatedTempRegs = findDuplicatedRegisters(tempRegisters);
      !duplicatedTempRegs.empty()) {
    srcMgr.PrintMessage(llvm::errs(),
                        llvm::SMLoc::getFromPointer(content.data()),
                        llvm::SourceMgr::DK_Error,
                        std::format("Duplicate temporary registers: {}",
                                    joinRegs(duplicatedTempRegs)));
    return utils::LogicalResult::failure();
  }

  if (!registerOptions->contains("int")) {
    srcMgr.PrintMessage(
        llvm::errs(), llvm::SMLoc::getFromPointer(content.data()),
        llvm::SourceMgr::DK_Error, "Missing 'int' register option");
    return utils::LogicalResult::failure();
  }

  auto intRegisters = registerOptions->find("int")->second;

  if (llvm::any_of(intRegisters, [](const as::Register &reg) {
        return reg.isFloatingPoint();
      })) {
    srcMgr.PrintMessage(llvm::errs(),
                        llvm::SMLoc::getFromPointer(content.data()),
                        llvm::SourceMgr::DK_Error,
                        "Integer registers must be integer registers");
    return utils::LogicalResult::failure();
  }

  if (auto duplicatedIntRegs = findDuplicatedRegisters(intRegisters);
      !duplicatedIntRegs.empty()) {
    srcMgr.PrintMessage(llvm::errs(),
                        llvm::SMLoc::getFromPointer(content.data()),
                        llvm::SourceMgr::DK_Error,
                        std::format("Duplicate integer registers: {}",
                                    joinRegs(duplicatedIntRegs)));
    return utils::LogicalResult::failure();
  }

  if (!registerOptions->contains("float")) {
    srcMgr.PrintMessage(
        llvm::errs(), llvm::SMLoc::getFromPointer(content.data()),
        llvm::SourceMgr::DK_Error, "Missing 'float' register option");
    return utils::LogicalResult::failure();
  }
  auto floatRegisters = registerOptions->find("float")->second;

  if (llvm::any_of(floatRegisters,
                   [](const as::Register &reg) { return reg.isInteger(); })) {
    srcMgr.PrintMessage(llvm::errs(),
                        llvm::SMLoc::getFromPointer(content.data()),
                        llvm::SourceMgr::DK_Error,
                        "Floating point registers must be floating point "
                        "registers");
    return utils::LogicalResult::failure();
  }

  if (auto duplicatedFloatRegs = findDuplicatedRegisters(floatRegisters);
      !duplicatedFloatRegs.empty()) {
    srcMgr.PrintMessage(llvm::errs(),
                        llvm::SMLoc::getFromPointer(content.data()),
                        llvm::SourceMgr::DK_Error,
                        std::format("Duplicate floating point registers: {}",
                                    joinRegs(duplicatedFloatRegs)));
    return utils::LogicalResult::failure();
  }

  setTempRegisters(tempRegisters);
  setRegistersForAllocate(intRegisters, floatRegisters);

  return utils::LogicalResult::success();
}

as::AnonymousRegisterStorage *TranslateContext::getAnonymousRegStorage() {
  return &impl->anonymousRegStorage;
}

TranslationRuleSet *TranslateContext::getTranslateRuleSet() {
  return &impl->translateRules;
}

} // namespace kecc
