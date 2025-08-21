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

template <typename T>
  requires std::is_enum_v<T>
struct EnumDenseMapInfo {
  using UnderlyingType = std::underlying_type_t<T>;

  static inline T getEmptyKey() {
    return static_cast<T>(DenseMapInfo<UnderlyingType>::getEmptyKey());
  }

  static inline T getTombstoneKey() {
    return static_cast<T>(DenseMapInfo<UnderlyingType>::getTombstoneKey());
  }
  static unsigned getHashValue(const T &Key) {
    return DenseMapInfo<UnderlyingType>::getHashValue(
        static_cast<UnderlyingType>(Key));
  }
  static bool isEqual(const T &LHS, const T &RHS) {
    return DenseMapInfo<UnderlyingType>::isEqual(
        static_cast<UnderlyingType>(LHS), static_cast<UnderlyingType>(RHS));
  }
};

} // namespace llvm
