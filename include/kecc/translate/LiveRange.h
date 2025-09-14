#ifndef KECC_TRANSLATE_LIVE_RANGE_H
#define KECC_TRANSLATE_LIVE_RANGE_H

#include "kecc/ir/Analysis.h"
#include "kecc/ir/IR.h"
#include "kecc/ir/Type.h"
#include "llvm/ADT/DenseMap.h"
#include <memory>

namespace kecc {

class LiveRangeStorage {
public:
  LiveRangeStorage(ir::Type type, ir::Function *func)
      : type(type), func(func) {}

  ir::Type getType() const { return type; }
  ir::Function *getFunction() const { return func; }

private:
  ir::Type type;
  ir::Function *func;
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
  ir::Function *getFunction() const { return storage->getFunction(); }

  friend llvm::hash_code hash_value(const LiveRange &lr) {
    return llvm::DenseMapInfo<const void *>::getHashValue(lr.getAsVoidPtr());
  }

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
    return hash_value(lr);
  }

  static bool isEqual(const kecc::LiveRange &l, const kecc::LiveRange &r) {
    return l == r;
  }
};

} // namespace llvm

#endif // KECC_TRANSLATE_LIVE_RANGE_H
