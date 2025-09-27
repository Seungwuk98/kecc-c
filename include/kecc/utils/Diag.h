#ifndef KECC_UTILS_DIAG_H
#define KECC_UTILS_DIAG_H

#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

namespace kecc::utils {

class DiagEngine {
public:
  DiagEngine(llvm::SourceMgr &srcMgr, llvm::raw_ostream *os = &llvm::errs())
      : srcMgr(srcMgr), os(os) {}

  void report(llvm::SMLoc loc, llvm::SourceMgr::DiagKind errKind,
              llvm::StringRef message) {
    srcMgr.PrintMessage(loc, errKind, message);
    errorCount += errKind == llvm::SourceMgr::DK_Error;
  }

  void report(llvm::SMLoc loc, llvm::SMRange range,
              llvm::SourceMgr::DiagKind errKind, llvm::StringRef message) {
    srcMgr.PrintMessage(loc, errKind, message, {range});
    errorCount += errKind == llvm::SourceMgr::DK_Error;
  }

  void report(llvm::SMRange range, llvm::SourceMgr::DiagKind errKind,
              llvm::StringRef message) {
    report(range.Start, range, errKind, message);
  }

  bool hasError() const { return errorCount > 0; }
  unsigned getErrorCount() const { return errorCount; }

  llvm::SourceMgr &getSourceMgr() { return srcMgr; }
  llvm::raw_ostream &getOS() { return *os; }

  void setOS(llvm::raw_ostream *newOS) { os = newOS; }

private:
  llvm::SourceMgr &srcMgr;
  llvm::raw_ostream *os;
  unsigned errorCount = 0;
};

} // namespace kecc::utils

#endif // KECC_UTILS_DIAG_H
