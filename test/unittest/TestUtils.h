#ifndef KECC_TEST_UTILS_H
#define KECC_TEST_UTILS_H

#include "doctest/doctest.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

namespace kecc {}

#define STR_EQ(a, b)                                                           \
  do {                                                                         \
    auto strA = llvm::StringRef(a).trim();                                     \
    auto strB = llvm::StringRef(b).trim();                                     \
    if (strA != strB) {                                                        \
      llvm::errs() << "Left:\n" << strA << "\n\n";                             \
      llvm::errs() << "Right:\n" << strB << '\n';                              \
      CHECK(false);                                                            \
    }                                                                          \
  } while (false)

#endif // KECC_TEST_UTILS_H
