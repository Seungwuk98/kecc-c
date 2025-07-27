#ifndef KECC_IR_CONTEXT_H
#define KECC_IR_CONTEXT_H

#include "kecc/utils/Diag.h"
#include "kecc/utils/MLIR.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"
#include <functional>
#include <limits>
#include <memory>
#include <vector>

namespace kecc::ir {

class AbstractInstruction;
class AbstractType;
class AbstractAttribute;
class AttributeImpl;
class TypeImpl;
class IRContext;

constexpr size_t BITS_OF_BYTE = 8;

template <typename T, typename... Args>
concept DetectGetKey = requires(T obj, Args &&...args) {
  {
    T::getKey(std::forward<Args>(args)...)
  } -> std::convertible_to<typename T::KeyTy>;
};

class TypeStorage {
private:
  struct FindKey {
    llvm::hash_code hashCode;
    std::function<bool(void *)> equals;
  };

  struct HashKey {
    llvm::hash_code hashCode;
    void *value;
  };

  struct HashKeyDenseMapInfo {
    static HashKey getEmptyKey() {
      return {llvm::DenseMapInfo<llvm::hash_code>::getEmptyKey(), nullptr};
    }

    static HashKey getTombstoneKey() {
      return {llvm::DenseMapInfo<llvm::hash_code>::getTombstoneKey(), nullptr};
    }

    static unsigned getHashValue(const HashKey &key) { return key.hashCode; }

    static unsigned getHashValue(const FindKey &key) { return key.hashCode; }

    static bool isEqual(const HashKey &lhs, const HashKey &rhs) {
      return lhs.hashCode == rhs.hashCode && lhs.value == rhs.value;
    }

    static bool isEqual(const FindKey &lhs, const HashKey &rhs) {
      return lhs.hashCode == rhs.hashCode && lhs.equals(rhs.value);
    }
  };

public:
  TypeStorage(IRContext *context) : context(context) {}

  template <typename TypeName, typename... Args>
  typename TypeName::ImplTy *getImplByArgs(Args &&...args) {
    using ImplTy = typename TypeName::ImplTy;
    using AbstractTable = typename TypeName::AbstractTableTy;

    if constexpr (std::is_same_v<ImplTy, typename ImplTy::ImplBase>) {
      static_assert(
          sizeof...(Args) == 0,
          "Singleton types should not be instantiated with arguments");

      auto it = singleTonTypes.find(TypeID::get<TypeName>());
      assert(it != singleTonTypes.end() &&
             "Singleton type not found in TypeStorage");
      return static_cast<ImplTy *>(it->second);
    } else {
      TypeID typeId = TypeID::get<TypeName>();

      auto key = getKey<ImplTy>(std::forward<Args>(args)...);
      auto hashValue =
          llvm::DenseMapInfo<typename ImplTy::KeyTy>::getHashValue(key);

      FindKey findKey{hashValue, [&key](void *value) -> bool {
                        return static_cast<ImplTy *>(value)->isEqual(key);
                      }};
      auto &cacheMap = typeCache[typeId];

      auto it = cacheMap.find_as(findKey);
      if (it != cacheMap.end()) {
        return static_cast<ImplTy *>(it->second);
      } else {
        auto *newImpl = ImplTy::create(this, key);
        auto *abstType = getAbstractTable<AbstractTable>(typeId);
        newImpl->setAbstractTable(abstType);
        HashKey hashKey{hashValue, newImpl};
        cacheMap.try_emplace(hashKey, newImpl);
        return newImpl;
      }
    }
  }

  template <typename ImplTy, typename... Args>
  typename ImplTy::KeyTy getKey(Args &&...args) {
    if constexpr (DetectGetKey<ImplTy, Args...>) {
      return ImplTy::getKey(std::forward<Args>(args)...);
    } else {
      return typename ImplTy::KeyTy(std::forward<Args>(args)...);
    }
  }

  template <typename T> T *allocate(size_t cnt = 1);

  llvm::StringRef copyString(llvm::StringRef str) {
    char *newStr = allocate<char>(str.size() + 1);
    std::copy(str.begin(), str.end(), newStr);
    newStr[str.size()] = '\0'; // Null-terminate the string
    return {newStr, str.size()};
  }

  template <typename T> llvm::ArrayRef<T> copyArray(llvm::ArrayRef<T> arr) {
    T *newArr = allocate<T>(arr.size());
    std::uninitialized_copy(arr.begin(), arr.end(), newArr);
    return llvm::ArrayRef<T>(newArr, arr.size());
  }

  template <typename T> T *getAbstractTable(TypeID typeId) {
    auto it = abstractTypes.find(typeId);
    if (it != abstractTypes.end()) {
      return static_cast<T *>(it->second);
    }
    return nullptr;
  }

  template <typename T> void registerType(AbstractType *abstractType) {
    using Impl = typename T::ImplTy;
    auto typeId = TypeID::get<T>();

    assert(abstractTypes.find(typeId) == abstractTypes.end() &&
           "Abstract type already registered");
    abstractTypes.try_emplace(typeId, abstractType);
    if constexpr (std::is_same_v<Impl, TypeImpl>) {
      auto *newImpl = new (allocate<Impl>()) Impl();
      newImpl->setAbstractTable(abstractType);
      singleTonTypes.try_emplace(typeId, newImpl);
    }
  }

  template <typename T> void registerAttr(AbstractAttribute *abstractAttr) {
    using Impl = typename T::ImplTy;
    auto typeId = TypeID::get<T>();
    assert(abstractTypes.find(typeId) == abstractTypes.end() &&
           "Abstract attribute already registered");
    abstractTypes.try_emplace(typeId, abstractAttr);
    if constexpr (std::is_same_v<Impl, AttributeImpl>) {
      auto *newImpl = new (allocate<Impl>()) Impl();
      newImpl->setAbstractTable(abstractAttr);
      singleTonTypes.try_emplace(typeId, newImpl);
    }
  }

  template <typename T> void registerInst(AbstractInstruction *abstractInst) {
    auto typeId = TypeID::get<T>();
    assert(abstractTypes.find(typeId) == abstractTypes.end() &&
           "Abstract instruction already registered");
    abstractTypes.try_emplace(typeId, abstractInst);
  }

private:
  llvm::DenseMap<TypeID, void *> abstractTypes;
  llvm::DenseMap<TypeID, llvm::DenseMap<HashKey, void *, HashKeyDenseMapInfo>>
      typeCache;
  llvm::DenseMap<TypeID, void *> singleTonTypes;
  IRContext *context;
};

class IRContext {
public:
  IRContext();
  ~IRContext();

  TypeStorage *getTypeStorage() const { return typeStorage.get(); }

  void *allocate(size_t size);

  template <typename T> T *allocate(size_t cnt) {
    return static_cast<T *>(sizeof(T) * cnt);
  }

  utils::DiagEngine &diag() { return diagEngine; }
  llvm::SourceMgr &getSourceMgr() { return srcMgr; }

  template <typename T> void registerType() {
    auto *abstType =
        GetAbstractTable<T, typename T::AbstractTableTy>::allocateAndGet(this);
    typeStorage->registerType<T>(abstType);
  }

  template <typename T> void registerAttr() {
    auto *abstAttr =
        GetAbstractTable<T, typename T::AbstractTableTy>::allocateAndGet(this);
    typeStorage->registerAttr<T>(abstAttr);
  }

  template <typename T> void registerInst() {
    auto *abstInst =
        GetAbstractTable<T, AbstractInstruction>::allocateAndGet(this);
    typeStorage->registerInst<T>(abstInst);
  }

  template <typename T> bool isRegisteredType() {
    TypeID typeId = TypeID::get<T>();
    return typeStorage->getAbstractTable<AbstractType>(typeId) != nullptr;
  }

  template <typename T> bool isRegisteredAttr() {
    TypeID typeId = TypeID::get<T>();
    return typeStorage->getAbstractTable<AbstractAttribute>(typeId) != nullptr;
  }

  template <typename T> bool isRegisteredInst() {
    TypeID typeId = TypeID::get<T>();
    return typeStorage->getAbstractTable<AbstractInstruction>(typeId) !=
           nullptr;
  }

  void setArchitectureBitSize(size_t bitSize) {
    architectrureBitSize = bitSize;
  }
  size_t getArchitectureBitSize() const { return architectrureBitSize; }

private:
  template <typename T, typename AbstractTableTy> struct GetAbstractTable {
    static AbstractTableTy *allocateAndGet(IRContext *context) {
      TypeID typeId = TypeID::get<T>();
      auto *abstTy = new (context->allocate(sizeof(AbstractTableTy)))
          AbstractTableTy(AbstractTableTy::template build<T>(context));
      return abstTy;
    }
  };

  std::unique_ptr<TypeStorage> typeStorage;
  llvm::SourceMgr srcMgr;
  utils::DiagEngine diagEngine;
  std::vector<void *> allocations;
  size_t architectrureBitSize = 64;
};

class RegisterId {
public:
  enum Kind {
    Arg,
    Temp,
    Alloc,
    Unresolved, // only for debuging
  };
  static constexpr int AllocBlockId = -1;

  RegisterId() = default;
  std::string toString() const;
  std::size_t getBlockId() const { return blockId; }
  Kind getKind() const { return kind; }
  std::size_t getRegId() const { return regId; }

  llvm::SMRange getRange() const { return range; }
  bool isArg() const { return kind == Kind::Arg; }
  bool isTemp() const { return kind == Kind::Temp; }
  bool isAlloc() const { return kind == Kind::Alloc; }

  static RegisterId arg(llvm::SMRange range, int blockId, std::size_t regId);

  static RegisterId temp(llvm::SMRange range, int blockId, std::size_t regId);

  static RegisterId alloc(llvm::SMRange range, std::size_t regId);

  static RegisterId unresolved(llvm::SMRange range, std::size_t regId);

private:
  friend llvm::DenseMapInfo<RegisterId>;

  RegisterId(llvm::SMRange range, std::size_t blockId, Kind kind,
             std::size_t regId)
      : range(range), blockId(blockId), kind(kind), regId(regId) {}

  llvm::SMRange range;
  int blockId;
  Kind kind;
  std::size_t regId;
};

template <typename T> T *TypeStorage::allocate(size_t cnt) {
  T *ptr = static_cast<T *>(this->context->allocate(sizeof(T) * cnt));
  return ptr;
}

} // namespace kecc::ir

namespace llvm {

template <> struct DenseMapInfo<kecc::ir::RegisterId> {
  static kecc::ir::RegisterId getEmptyKey() {
    return kecc::ir::RegisterId({}, -1, kecc::ir::RegisterId::Kind::Alloc,
                                std::numeric_limits<std::size_t>::max());
  }

  static kecc::ir::RegisterId getTombstoneKey() {
    return kecc::ir::RegisterId({}, -1, kecc::ir::RegisterId::Kind::Alloc,
                                std::numeric_limits<std::size_t>::max() - 1);
  }

  static unsigned getHashValue(const kecc::ir::RegisterId &id) {
    return llvm::hash_combine(id.getBlockId(), id.getKind(), id.getRegId());
  }

  static bool isEqual(const kecc::ir::RegisterId &lhs,
                      const kecc::ir::RegisterId &rhs) {
    return lhs.getBlockId() == rhs.getBlockId() &&
           lhs.getKind() == rhs.getKind() && lhs.getRegId() == rhs.getRegId();
  }
};

} // namespace llvm

#endif // KECC_IR_CONTEXT_H
