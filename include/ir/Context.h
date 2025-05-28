#ifndef KECC_IR_CONTEXT_H
#define KECC_IR_CONTEXT_H

#include "utils/TypeId.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/StringRef.h"
#include <functional>
#include <memory>
#include <vector>

namespace kecc::ir {

class AbstractType;
class TypeImpl;

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

  struct FindKeyDenseMapInfo {
    static FindKey getEmptyKey() {
      return {llvm::DenseMapInfo<llvm::hash_code>::getEmptyKey(), nullptr};
    }

    static FindKey getTombstoneKey() {
      return {llvm::DenseMapInfo<llvm::hash_code>::getTombstoneKey(), nullptr};
    }

    static unsigned getHashValue(const FindKey &key) { return key.hashCode; }

    static bool isEqual(const FindKey &lhs, const HashKey &rhs) {
      return lhs.hashCode == rhs.hashCode && lhs.equals(rhs.value);
    }
  };

  struct HashKeyDenseMapInfo {
    static HashKey getEmptyKey() {
      return {llvm::DenseMapInfo<llvm::hash_code>::getEmptyKey(), nullptr};
    }

    static HashKey getTombstoneKey() {
      return {llvm::DenseMapInfo<llvm::hash_code>::getTombstoneKey(), nullptr};
    }

    static unsigned getHashValue(const HashKey &key) { return key.hashCode; }

    static bool isEqual(const HashKey &lhs, const HashKey &rhs) {
      return lhs.hashCode == rhs.hashCode && lhs.value == rhs.value;
    }

    static bool isEqual(const FindKey &lhs, const HashKey &rhs) {
      return lhs.hashCode == rhs.hashCode && lhs.equals(rhs.value);
    }
  };

public:
  ~TypeStorage() {
    for (auto *ptr : allocations)
      free(ptr);
  }

  template <typename TypeName, typename... Args>
  typename TypeName::Impl *getImplByArgs(Args &&...args) {
    using Impl = typename TypeName::Impl;

    if constexpr (std::is_same_v<Impl, TypeImpl>) {
      static_assert(
          sizeof...(Args) == 0,
          "Singleton types should not be instantiated with arguments");

      auto it = singleTonTypes.find(utils::getId<TypeName>());
      assert(it != singleTonTypes.end() &&
             "Singleton type not found in TypeStorage");
      return static_cast<Impl *>(it->second);
    }

    utils::TypeId typeId = utils::getId<TypeName>;

    auto key = Impl::KeyTy(std::forward<Args>(args)...);
    auto hashValue =
        llvm::DenseMapInfo<typename Impl::KeyTy>::getHashValue(key);

    FindKey findKey{hashValue, [](void *value) -> bool {
                      return static_cast<Impl *>(value)->isEqual(key);
                    }};
    auto &cacheMap = typeCache[typeId];

    auto it = cacheMap.find_as(findKey);
    if (it != cacheMap.end()) {
      return static_cast<Impl *>(it->second);
    } else {
      auto *newImpl = Impl::create(this, std::forward<Args>(args)...);
      auto *abstType = getAbstractType(typeId);
      newImpl->setAbstractType(abstType);
      HashKey hashKey{hashValue, newImpl};
      cacheMap.try_emplace(hashKey, newImpl);
      return newImpl;
    }
  }

  template <typename T> T *allocate(size_t cnt = 1) {
    T *ptr = static_cast<T *>(llvm::safe_malloc(sizeof(T) * cnt));
    allocations.emplace_back(ptr);
    return ptr;
  }

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

  AbstractType *getAbstractType(utils::TypeId typeId) {
    auto it = abstractTypes.find(typeId);
    if (it != abstractTypes.end()) {
      return it->second;
    }
    return nullptr;
  }

  void registerAbstractType(utils::TypeId typeId, AbstractType *abstractType) {
    assert(abstractTypes.find(typeId) == abstractTypes.end() &&
           "Abstract type already registered");
    abstractTypes.try_emplace(typeId, abstractType);
  }

private:
  llvm::DenseMap<utils::TypeId, AbstractType *> abstractTypes;
  llvm::DenseMap<utils::TypeId,
                 llvm::DenseMap<HashKey, void *, HashKeyDenseMapInfo>>
      typeCache;
  llvm::DenseMap<utils::TypeId, void *> singleTonTypes;
  std::vector<void *> allocations;
};

class IRContext {
public:
  IRContext();

  TypeStorage *getTypeStorage() const { return typeStorage.get(); }

private:
  std::unique_ptr<TypeStorage> typeStorage;
};

} // namespace kecc::ir

#endif // KECC_IR_CONTEXT_H
