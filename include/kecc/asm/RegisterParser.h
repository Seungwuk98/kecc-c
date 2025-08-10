#ifndef KECC_ASM_REGISTER_PARSER_H
#define KECC_ASM_REGISTER_PARSER_H

#include "kecc/asm/Register.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

namespace kecc::as {

class RegisterParser {
public:
  RegisterParser(AnonymousRegisterStorage *storage, llvm::SourceMgr &srcMgr,
                 llvm::StringRef buffer)
      : storage(storage), srcMgr(srcMgr), buffer(buffer), cursor(0),
        savedCursor(0) {}

  // registers = '{' register (',' register)* '}'
  std::optional<llvm::SmallVector<Register>> parseRegisters();

  // register := builtin | anonymous
  // builtin := x[0-9]+ | f[0-9]+ | zero | ra | sp | gp | tp | t[0-9]+ | s[0-9]+
  //            | a[0-9]+ | ft[0-9]+ | fs[0-9]+ | fa[0-9]+
  // anonymous := identifier '(' ('int' | 'float') ',' ('caller-save' |
  // 'callee-saved') ')'
  std::optional<Register> parseRegister();

  // anonymous-info := '(' ('int' | 'float') ',' ('caller-save' |
  //                   'callee-save') ')'
  std::optional<Register> parseAnonymousRegister(llvm::StringRef name);

  // key-values := identifier '=' registers
  // register-map := key-values (',' key-values)*
  std::optional<llvm::MapVector<llvm::StringRef, llvm::SmallVector<Register>>>
  parseRegiserOptions();

private:
  std::optional<std::pair<llvm::StringRef, llvm::SmallVector<Register>>>
  parseRegisterOption();

  void skipWhitespace();

  llvm::StringRef lexIdentifier();

  // Returns false if the next character is a comma and consumes it.
  bool consume(char ch);

  char peek() const;
  char advance();

  void report(llvm::SMLoc loc, llvm::StringRef message) {
    srcMgr.PrintMessage(llvm::errs(), loc, llvm::SourceMgr::DK_Error, message);
  }

  llvm::SMLoc currLoc() const {
    return llvm::SMLoc::getFromPointer(buffer.data() + cursor);
  }

  AnonymousRegisterStorage *storage;
  llvm::SourceMgr &srcMgr;
  llvm::StringRef buffer;
  size_t cursor = 0;
  size_t savedCursor = 0;
};

} // namespace kecc::as

#endif // KECC_ASM_REGISTER_PARSER_H
