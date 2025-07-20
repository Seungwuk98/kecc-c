#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/DenseMapInfo.h"

namespace llvm {
template <> struct DenseMapInfo<APFloat> {
  static inline APFloat getEmptyKey() { return APFloat(APFloat::Bogus(), 1); }
  static inline APFloat getTombstoneKey() {
    return APFloat(APFloat::Bogus(), 2);
  }

  static unsigned getHashValue(const APFloat &Key) {
    return static_cast<unsigned>(hash_value(Key));
  }

  static bool isEqual(const APFloat &LHS, const APFloat &RHS) {
    return LHS.bitwiseIsEqual(RHS);
  }
};
} // namespace llvm
