#ifndef KECC_TRANSLATE_LIVE_RANGE_H
#define KECC_TRANSLATE_LIVE_RANGE_H

#include "kecc/ir/Analysis.h"
#include "llvm/ADT/DenseMap.h"
#include <memory>

namespace kecc {
class LiveRange {
public:
  LiveRange() : ptr(nullptr) {}
  LiveRange(const void *ptr) : ptr(ptr) {}
  static LiveRange getFromVoidPtr(const void *ptr) { return LiveRange(ptr); }

  const void *getPtr() const { return ptr; }

  bool operator==(const LiveRange &other) const { return ptr == other.ptr; }
  bool operator!=(const LiveRange &other) const { return ptr != other.ptr; }
  operator bool() const { return ptr != nullptr; }

private:
  const void *ptr;
};

} // namespace kecc

namespace llvm {

template <> struct DenseMapInfo<kecc::LiveRange> {
  static kecc::LiveRange getEmptyKey() {
    return kecc::LiveRange(llvm::DenseMapInfo<const void *>::getEmptyKey());
  }

  static kecc::LiveRange getTombstoneKey() {
    return kecc::LiveRange(llvm::DenseMapInfo<const void *>::getTombstoneKey());
  }

  static unsigned getHashValue(const kecc::LiveRange &lr) {
    return llvm::DenseMapInfo<const void *>::getHashValue(lr.getPtr());
  }

  static bool isEqual(const kecc::LiveRange &l, const kecc::LiveRange &r) {
    return l == r;
  }
};

} // namespace llvm

#endif // KECC_TRANSLATE_LIVE_RANGE_H
