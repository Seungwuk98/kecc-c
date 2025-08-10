#include "kecc/asm/RegisterParser.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Error.h"
#include <cstdio>
#include <format>

namespace kecc::as {

void RegisterParser::skipWhitespace() {
  while (cursor < buffer.size() && std::isspace(buffer[cursor]))
    ++cursor;
}

llvm::StringRef RegisterParser::lexIdentifier() {
  savedCursor = cursor;
  char p = peek();
  if (!std::isalpha(p) && p != '_')
    return {};
  advance();

  while (cursor < buffer.size() && (std::isalnum(peek()) || peek() == '_'))
    advance();

  return buffer.slice(savedCursor, cursor);
}

bool RegisterParser::consume(char ch) {
  if (peek() == ch) {
    advance();
    return false;
  }
  return true;
}

char RegisterParser::peek() const {
  if (cursor >= buffer.size())
    return EOF;
  return buffer[cursor];
}

char RegisterParser::advance() {
  if (cursor >= buffer.size())
    return EOF;
  return buffer[cursor++];
}

std::optional<llvm::SmallVector<Register>> RegisterParser::parseRegisters() {
  llvm::SmallVector<Register> registers;

  skipWhitespace();
  if (consume('{')) {
    report(currLoc(), "Expected '{'");
    return std::nullopt;
  }

  skipWhitespace();
  if (peek() == '}') {
    advance();
    return llvm::SmallVector<Register>{};
  }

  auto reg = parseRegister();
  if (!reg)
    return std::nullopt;

  registers.emplace_back(*reg);
  skipWhitespace();
  while (peek() == ',') {
    advance();
    reg = parseRegister();
    if (!reg)
      return std::nullopt;
    registers.emplace_back(*reg);
    skipWhitespace();
  }

  if (consume('}')) {
    report(currLoc(), "Expected '}'");
    return std::nullopt;
  }

  return registers;
}

std::optional<Register> RegisterParser::parseRegister() {
  skipWhitespace();
  llvm::StringRef regName = lexIdentifier();
  if (regName.empty()) {
    report(currLoc(), "Expected register name");
    return std::nullopt;
  }
  std::optional<Register> retReg =
      llvm::StringSwitch<std::optional<Register>>(regName)
#define CASE(Name) Case(#Name, Register::Name())
          .CASE(zero)
          .CASE(ra)
          .CASE(sp)
          .CASE(gp)
          .CASE(tp)
          .CASE(t0)
          .CASE(t1)
          .CASE(t2)
          .CASE(t3)
          .CASE(t4)
          .CASE(t5)
          .CASE(t6)
          .CASE(s0)
          .CASE(s1)
          .CASE(s2)
          .CASE(s3)
          .CASE(s4)
          .CASE(s5)
          .CASE(s6)
          .CASE(s7)
          .CASE(s8)
          .CASE(s9)
          .CASE(s10)
          .CASE(s11)
          .CASE(a0)
          .CASE(a1)
          .CASE(a2)
          .CASE(a3)
          .CASE(a4)
          .CASE(a5)
          .CASE(a6)
          .CASE(a7)
          .CASE(ft0)
          .CASE(ft1)
          .CASE(ft2)
          .CASE(ft3)
          .CASE(ft4)
          .CASE(ft5)
          .CASE(ft6)
          .CASE(ft7)
          .CASE(ft8)
          .CASE(ft9)
          .CASE(ft10)
          .CASE(ft11)
          .CASE(fs0)
          .CASE(fs1)
          .CASE(fs2)
          .CASE(fs3)
          .CASE(fs4)
          .CASE(fs5)
          .CASE(fs6)
          .CASE(fs7)
          .CASE(fs8)
          .CASE(fs9)
          .CASE(fs10)
          .CASE(fs11)
          .CASE(fa0)
          .CASE(fa1)
          .CASE(fa2)
          .CASE(fa3)
          .CASE(fa4)
          .CASE(fa5)
          .CASE(fa6)
          .CASE(fa7)
#undef CASE
#define CASE_X(Index) Case("x" #Index, Register::getX(Index))

          .CASE_X(0)
          .CASE_X(1)
          .CASE_X(2)
          .CASE_X(3)
          .CASE_X(4)
          .CASE_X(5)
          .CASE_X(6)
          .CASE_X(7)
          .CASE_X(8)
          .CASE_X(9)
          .CASE_X(10)
          .CASE_X(11)
          .CASE_X(12)
          .CASE_X(13)
          .CASE_X(14)
          .CASE_X(15)
          .CASE_X(16)
          .CASE_X(17)
          .CASE_X(18)
          .CASE_X(19)
          .CASE_X(20)
          .CASE_X(21)
          .CASE_X(22)
          .CASE_X(23)
          .CASE_X(24)
          .CASE_X(25)
          .CASE_X(26)
          .CASE_X(27)
          .CASE_X(28)
          .CASE_X(29)
          .CASE_X(30)
          .CASE_X(31)

#undef CASE_X
#define CASE_F(Index) Case("f" #Index, Register::getF(Index))

          .CASE_F(0)
          .CASE_F(1)
          .CASE_F(2)
          .CASE_F(3)
          .CASE_F(4)
          .CASE_F(5)
          .CASE_F(6)
          .CASE_F(7)
          .CASE_F(8)
          .CASE_F(9)
          .CASE_F(10)
          .CASE_F(11)
          .CASE_F(12)
          .CASE_F(13)
          .CASE_F(14)
          .CASE_F(15)
          .CASE_F(16)
          .CASE_F(17)
          .CASE_F(18)
          .CASE_F(19)
          .CASE_F(20)
          .CASE_F(21)
          .CASE_F(22)
          .CASE_F(23)
          .CASE_F(24)
          .CASE_F(25)
          .CASE_F(26)
          .CASE_F(27)
          .CASE_F(28)
          .CASE_F(29)
          .CASE_F(30)
          .CASE_F(31)

#undef CASE_F
          .Default(std::nullopt);

  if (peek() == '(') {
    if (retReg) {
      report(currLoc(),
             std::format("Register '{}' cannot be an anonymous register",
                         regName.str()));
      return std::nullopt;
    }

    return parseAnonymousRegister(regName);
  }

  if (!retReg) {
    report(currLoc(), ("Invalid register name: " + regName).str());
    return std::nullopt;
  }

  return retReg;
}

std::optional<Register>
RegisterParser::parseAnonymousRegister(llvm::StringRef name) {
  if (consume('(')) {
    report(currLoc(), "Expected '('");
    return std::nullopt;
  }

  auto intOrFloat = lexIdentifier();
  if (intOrFloat.empty()) {
    report(currLoc(), "Expected 'int' or 'float'");
    return std::nullopt;
  }

  skipWhitespace();

  RegisterType regType;
  if (intOrFloat == "int")
    regType = RegisterType::Integer;
  else if (intOrFloat == "float")
    regType = RegisterType::FloatingPoint;
  else {
    report(currLoc(), "Expected 'int' or 'float'");
    return std::nullopt;
  }

  skipWhitespace();

  if (consume(',')) {
    report(currLoc(), "Expected ','");
    return std::nullopt;
  }

  skipWhitespace();

  llvm::StringRef callerOrCallee = lexIdentifier();
  CallingConvension callingConvention;
  if (callerOrCallee == "caller")
    callingConvention = CallingConvension::CallerSave;
  else if (callerOrCallee == "callee")
    callingConvention = CallingConvension::CalleeSave;
  else {
    report(currLoc(), "Expected 'caller-save' or 'callee-save'");
    return std::nullopt;
  }
  if (consume('-')) {
    report(currLoc(), "Expected '-' after 'caller' or 'callee'");
    return std::nullopt;
  }
  llvm::StringRef lexSave = lexIdentifier();
  if (lexSave != "save") {
    report(currLoc(), "Expected 'save' after 'caller-' or 'callee-'");
    return std::nullopt;
  }

  skipWhitespace();
  if (consume(')')) {
    report(currLoc(), "Expected ')'");
    return std::nullopt;
  }

  if (Register::definedAnonymousRegister(storage, name)) {
    report(currLoc(), std::format("Anonymous register '{}' is already defined",
                                  name.str()));
    return std::nullopt;
  }

  auto anonymousReg = Register::createAnonymousRegister(
      storage, regType, callingConvention, name);
  return anonymousReg;
}

std::optional<llvm::MapVector<llvm::StringRef, llvm::SmallVector<Register>>>
RegisterParser::parseRegiserOptions() {
  llvm::MapVector<llvm::StringRef, llvm::SmallVector<Register>> options;

  auto regOptionOpt = parseRegisterOption();
  if (!regOptionOpt)
    return std::nullopt;

  auto [_, inserted] =
      options.try_emplace(regOptionOpt->first, regOptionOpt->second);
  if (!inserted) {
    report(currLoc(), std::format("Duplicate register option key: '{}'",
                                  regOptionOpt->first.str()));
    return std::nullopt;
  }

  skipWhitespace();
  while (peek() == ',') {
    advance();
    regOptionOpt = parseRegisterOption();
    if (!regOptionOpt)
      return std::nullopt;

    auto [_, inserted] =
        options.try_emplace(regOptionOpt->first, regOptionOpt->second);
    if (!inserted) {
      report(currLoc(), std::format("Duplicate register option key: '{}'",
                                    regOptionOpt->first.str()));
      return std::nullopt;
    }
    skipWhitespace();
  }

  return options;
}

std::optional<std::pair<llvm::StringRef, llvm::SmallVector<Register>>>
RegisterParser::parseRegisterOption() {
  skipWhitespace();
  llvm::StringRef key = lexIdentifier();
  if (key.empty()) {
    report(currLoc(), "Expected register option key");
    return std::nullopt;
  }

  if (consume('=')) {
    report(currLoc(), "Expected '=' after register option key");
    return std::nullopt;
  }

  auto registers = parseRegisters();
  if (!registers)
    return std::nullopt;

  return std::pair(key, *registers);
}

} // namespace kecc::as
