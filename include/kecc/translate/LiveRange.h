#ifndef KECC_TRANSLATE_LIVE_RANGE_H
#define KECC_TRANSLATE_LIVE_RANGE_H

#include "kecc/ir/Analysis.h"
#include "kecc/ir/Type.h"
#include "llvm/ADT/DenseMap.h"
#include <memory>

namespace kecc {

class LiveRangeStorage {
public:
  LiveRangeStorage(ir::Type type) : type(type) {}
  ir::Type getType() const { return type; }

private:
  ir::Type type;
};

class LiveRange {
public:
  LiveRange() : storage(nullptr) {}
  LiveRange(const LiveRangeStorage *ptr) : storage(ptr) {}

  static LiveRange getFromVoidPtr(const void *ptr) {
    return LiveRange(static_cast<const LiveRangeStorage *>(ptr));
  }

  const void *getAsVoidPtr() const { return storage; }

  bool operator==(const LiveRange &other) const {
    return storage == other.storage;
  }
  bool operator!=(const LiveRange &other) const {
    return storage != other.storage;
  }
  operator bool() const { return storage != nullptr; }

  ir::Type getType() const { return storage->getType(); }

private:
  const LiveRangeStorage *storage;
};

} // namespace kecc

namespace llvm {

template <> struct DenseMapInfo<kecc::LiveRange> {
  static kecc::LiveRange getEmptyKey() {
    return kecc::LiveRange::getFromVoidPtr(
        llvm::DenseMapInfo<const void *>::getEmptyKey());
  }

  static kecc::LiveRange getTombstoneKey() {
    return kecc::LiveRange::getFromVoidPtr(
        llvm::DenseMapInfo<const void *>::getTombstoneKey());
  }

  static unsigned getHashValue(const kecc::LiveRange &lr) {
    return llvm::DenseMapInfo<const void *>::getHashValue(lr.getAsVoidPtr());
  }

  static bool isEqual(const kecc::LiveRange &l, const kecc::LiveRange &r) {
    return l == r;
  }
};

} // namespace llvm

#endif // KECC_TRANSLATE_LIVE_RANGE_H
