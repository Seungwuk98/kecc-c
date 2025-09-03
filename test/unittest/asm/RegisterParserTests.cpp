#include "TestUtils.h"
#include "kecc/asm/RegisterParser.h"
#include "llvm/Support/SourceMgr.h"
#include <format>

namespace kecc::as {

std::optional<Register> parseRegister(llvm::SourceMgr &srcMgr,
                                      AnonymousRegisterStorage *storage,
                                      llvm::StringRef regStr) {
  auto memBuffer = llvm::MemoryBuffer::getMemBuffer(regStr, "<register>");
  srcMgr.AddNewSourceBuffer(std::move(memBuffer), llvm::SMLoc());
  RegisterParser parser(storage, srcMgr, regStr);

  auto reg = parser.parseRegister();
  if (!reg) {
    FAIL(std::format("Failed to parse register: {}", regStr.str()));
    return std::nullopt;
  }

  return reg;
}

std::optional<llvm::SmallVector<Register>>
parseRegisters(llvm::SourceMgr &srcMgr, AnonymousRegisterStorage *storage,
               llvm::StringRef regStr) {
  auto memBuffer = llvm::MemoryBuffer::getMemBuffer(regStr, "<registers>");
  srcMgr.AddNewSourceBuffer(std::move(memBuffer), llvm::SMLoc());
  RegisterParser parser(storage, srcMgr, regStr);
  auto regs = parser.parseRegisters();
  if (!regs) {
    FAIL(std::format("Failed to parse registers: {}", regStr.str()));
    return std::nullopt;
  }

  return regs;
}

std::optional<llvm::MapVector<llvm::StringRef, llvm::SmallVector<Register>>>
parseRegisterOptions(llvm::SourceMgr &srcMgr, AnonymousRegisterStorage *storage,
                     llvm::StringRef regStr) {
  auto memBuffer =
      llvm::MemoryBuffer::getMemBuffer(regStr, "<register options>");
  srcMgr.AddNewSourceBuffer(std::move(memBuffer), llvm::SMLoc());
  RegisterParser parser(storage, srcMgr, regStr);
  auto options = parser.parseRegiserOptions();
  if (!options) {
    FAIL(std::format("Failed to parse register options: {}", regStr.str()));
    return std::nullopt;
  }
  return options;
}

TEST_CASE("Register Parser Tests") {
  llvm::SourceMgr srcMgr;
  AnonymousRegisterStorage storage;

  SUBCASE("simple register") {
    auto reg = parseRegister(srcMgr, &storage, "a0");
    REQUIRE(reg);

    CHECK(*reg == Register::a0());
  }

  SUBCASE("simple register - 2") {
    auto reg = parseRegister(srcMgr, &storage, "fa0");
    REQUIRE(reg);

    CHECK(*reg == Register::fa0());
  }

  SUBCASE("anonymous register") {
    auto reg = parseRegister(srcMgr, &storage, "anonymous(int, caller-save)");
    REQUIRE(reg);
    auto anon = Register::getAnonymousRegister(&storage, "anonymous");
    CHECK(*reg == anon);
    CHECK(anon.isCallerSaved());
    CHECK(anon.isInteger());
    CHECK(anon.toString() == "anonymous");
  }

  SUBCASE("registers") {
    auto regs = parseRegisters(srcMgr, &storage, "{a0, a1, a2, a3}");
    REQUIRE(regs);

    CHECK(regs->size() == 4);
    CHECK((*regs)[0] == Register::a0());
    CHECK((*regs)[1] == Register::a1());
    CHECK((*regs)[2] == Register::a2());
    CHECK((*regs)[3] == Register::a3());
  }

  SUBCASE("register option") {
    auto regOptions = parseRegisterOptions(srcMgr, &storage, R"(
                                           temp={t0, t1},
                                           int={a0, a1, a2, a3},
                                           float={fa0, fa1, fa2, fa3}
                                           )");
    REQUIRE(regOptions);
    CHECK(regOptions->size() == 3);
    CHECK(regOptions->contains("temp"));
    CHECK(regOptions->contains("int"));
    CHECK(regOptions->contains("float"));

    const auto &tempRegs = (*regOptions)["temp"];
    CHECK(tempRegs.size() == 2);
    CHECK(tempRegs[0] == Register::t0());
    CHECK(tempRegs[1] == Register::t1());

    const auto &intRegs = (*regOptions)["int"];
    CHECK(intRegs.size() == 4);
    CHECK(intRegs[0] == Register::a0());
    CHECK(intRegs[1] == Register::a1());
    CHECK(intRegs[2] == Register::a2());
    CHECK(intRegs[3] == Register::a3());

    const auto &floatRegs = (*regOptions)["float"];
    CHECK(floatRegs.size() == 4);
    CHECK(floatRegs[0] == Register::fa0());
    CHECK(floatRegs[1] == Register::fa1());
    CHECK(floatRegs[2] == Register::fa2());
    CHECK(floatRegs[3] == Register::fa3());
  }
}

} // namespace kecc::as
