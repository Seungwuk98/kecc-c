#ifndef KECC_IR_TYPES_H
#define KECC_IR_TYPES_H

#include "ir/Context.h"
#include "utils/TypeId.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

namespace kecc::ir {

class TypeImpl;

class Type {
public:
  Type() : impl(nullptr) {}
  Type(TypeImpl *impl) : impl(impl) {}

  bool operator==(const Type &other) const { return impl == other.impl; }
  bool operator!=(const Type &other) const { return impl != other.impl; }
  operator bool() const { return impl != nullptr; }

  template <typename U> bool isa() const { return llvm::isa<U>(impl); }
  template <typename U> U cast() const { return llvm::cast<U>(impl); }
  template <typename U> U dyn_cast() const { return llvm::dyn_cast<U>(impl); }
  template <typename U> U dyn_cast_or_null() const {
    return llvm::dyn_cast_or_null<U>(impl);
  }

  TypeImpl *getImpl() const { return impl; }

  utils::TypeId getId() const;

  void print(llvm::raw_ostream &os = llvm::errs()) const;
  std::string toString() const;

private:
  TypeImpl *impl;
};

class AbstractType {
public:
  using PrintFn = std::function<void(Type, llvm::raw_ostream &)>;

  utils::TypeId getId() const { return id; }
  IRContext *getContext() const { return context; }

  template <typename T> static AbstractType build(IRContext *context) {
    utils::TypeId typeId = utils::getId<T>;
    return AbstractType(typeId, context, T::getPrintFn());
  }

private:
  AbstractType(utils::TypeId id, IRContext *context, PrintFn printFn)
      : id(id), context(context), printFn(std::move(printFn)) {}

  utils::TypeId id;
  IRContext *context;
  PrintFn printFn;
};

class TypeImpl {
public:
  TypeImpl() = default;
  utils::TypeId getId() const { return abstractType->getId(); }
  IRContext *getContext() const { return abstractType->getContext(); }

private:
  friend class TypeStorage;
  void setAbstractType(AbstractType *abstractType) {
    this->abstractType = abstractType;
  }
  AbstractType *abstractType;
};

struct TypeBuilder {
  template <typename T, typename... Args>
  static T get(IRContext *context, Args &&...args) {
    TypeStorage *storage = context->getTypeStorage();
    return storage->getImplByArgs(std::forward<Args>(args)...);
  }

  template <typename T>
  static AbstractType *getAbstractType(IRContext *context) {
    utils::TypeId typeId = utils::getId<T>;
    TypeStorage *storage = context->getTypeStorage();
    return storage->getAbstractType(typeId);
  }
};

template <typename T>
concept HasVerify = requires(T) {
  { T::verify() } -> std::convertible_to<bool>;
};

template <typename ConcreteType, typename ParentType, typename ImplTy>
struct TypeTemplate : public ParentType {
  using Impl = ImplTy;
  using Base = TypeTemplate<ConcreteType, ParentType, Impl>;

  ImplTy *getImpl() const {
    return static_cast<ImplTy *>(ParentType::getImpl());
  }

  template <typename... Args>
  static ConcreteType get(IRContext *context, Args &&...args) {
    if constexpr (HasVerify<ConcreteType>) {
      if (!ConcreteType::verify(std::forward<Args>(args)...))
        return nullptr;
    }
    return TypeBuilder::get<ConcreteType>(context, std::forward<Args>(args)...);
  }

  template <typename T>
    requires(std::is_convertible_v<T, Type>)
  static bool classof(const Type &type) {
    return type.getId() == utils::getId<ConcreteType>;
  }

  static auto getPrintFn() {
    return [](Type type, llvm::raw_ostream &os) {
      ConcreteType::printer(type.cast<ConcreteType>(), os);
    };
  }
};

template <typename KeyType> struct TypeImplTemplate : public TypeImpl {
  using KeyTy = KeyType;

  bool isEqual(const KeyTy &otherKey) const { return key == otherKey; }

protected:
  TypeImplTemplate(KeyTy key) : key(key) {}
  const KeyTy &getKey() const { return key; }

private:
  KeyTy key;
};

} // namespace kecc::ir

namespace llvm {
template <> struct DenseMapInfo<kecc::ir::Type> {
  static kecc::ir::Type getEmptyKey() { return nullptr; }

  static kecc::ir::Type getTombstoneKey() {
    return DenseMapInfo<kecc::ir::TypeImpl *>::getTombstoneKey();
  }

  static unsigned getHashValue(const kecc::ir::Type &type) {
    return DenseMapInfo<kecc::ir::TypeImpl *>::getHashValue(type.getImpl());
  }

  static bool isEqual(const kecc::ir::Type &lhs, const kecc::ir::Type &rhs) {
    return lhs == rhs;
  }
};

template <typename To, typename From>
struct CastInfo<To, From,
                std::enable_if_t<
                    std::is_same_v<kecc::ir::Type, std::remove_const_t<From>> ||
                    std::is_base_of_v<kecc::ir::Type, From>>>
    : NullableValueCastFailed<To>,
      DefaultDoCastIfPossible<To, From, CastInfo<To, From>> {
  static inline bool isPossible(kecc::ir::Type ty) {
    if constexpr (std::is_base_of_v<To, From>) {
      return true;
    } else {
      return To::classof(ty);
    };
  }
  static inline To doCast(kecc::ir::Type ty) { return To(ty.getImpl()); }
};
} // namespace llvm

#endif // KECC_IR_TYPES_H
