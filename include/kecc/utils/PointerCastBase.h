#ifndef KECC_UTILS_POINTER_CAST_BASE_H
#define KECC_UTILS_POINTER_CAST_BASE_H

#include "llvm/Support/Casting.h"

namespace kecc::utils {
template <typename T> struct PointerCastBase {
  template <typename... Us> bool isa() const {
    return llvm::isa<Us...>(derived());
  }
  template <typename U> U *cast() { return llvm::cast<U>(derived()); }
  template <typename U> const U *cast() const {
    return llvm::cast<U>(derived());
  }
  template <typename U> U *dyn_cast() { return llvm::dyn_cast<U>(derived()); }
  template <typename U> const U *dyn_cast() const {
    return llvm::dyn_cast<U>(derived());
  }

private:
  T *derived() { return static_cast<T *>(this); }
  const T *derived() const { return static_cast<const T *>(this); }
};

} // namespace kecc::utils

#endif // KECC_UTILS_POINTER_CAST_BASE_H
