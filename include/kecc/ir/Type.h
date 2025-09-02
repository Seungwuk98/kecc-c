#ifndef KECC_IR_TYPES_H
#define KECC_IR_TYPES_H

#include "kecc/ir/Context.h"
#include "kecc/ir/TypeAttributeSupport.h"
#include "kecc/ir/WalkSupport.h"
#include "kecc/utils/MLIR.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

namespace kecc::ir {

class Type;
class TypeImpl;
struct TypeBuilder;
class AbstractType;

class Type {
public:
  template <typename ConcreteType, typename ParentType, typename ImplType>
  using Base = TypeAttrTemplate<ConcreteType, ParentType, Type, TypeBuilder,
                                ImplType, AbstractType>;
  Type() : impl(nullptr) {}
  Type(TypeImpl *impl) : impl(impl) {}
  Type(const Type &other) : impl(other.impl) {}

  bool operator==(const Type &other) const { return impl == other.impl; }
  bool operator!=(const Type &other) const { return impl != other.impl; }
  operator bool() const { return impl != nullptr; }

  template <typename... U> bool isa() const { return llvm::isa<U...>(*this); }
  template <typename U> U cast() const { return llvm::cast<U>(*this); }
  template <typename U> U dyn_cast() const { return llvm::dyn_cast<U>(*this); }
  template <typename U> U dyn_cast_or_null() const {
    return llvm::dyn_cast_or_null<U>(*this);
  }

  TypeImpl *getImpl() const { return impl; }

  TypeID getId() const;

  void print(llvm::raw_ostream &os = llvm::errs()) const;
  std::string toString() const;

  friend llvm::hash_code hash_value(const Type &type) {
    return llvm::hash_value(type.getImpl());
  }

  Type constCanonicalize() const;

  IRContext *getContext() const;

  std::pair<size_t, size_t> getSizeAndAlign(const StructSizeMap &sizeMap) const;

  template <TypeWalker::Order order = TypeWalker::Order::PreOrder,
            typename... WalkFns>
  WalkResult walk(WalkFns &&...walkFns) const {
    TypeWalker walker(order);
    (walker.addWalkFn(std::forward<WalkFns>(walkFns)), ...);
    return walker.walk(*this);
  }

  template <typename... ReplaceFns>
  Type replace(ReplaceFns &&...replaceFn) const {
    TypeReplacer replacer;
    (replacer.addReplaceFn(std::forward<ReplaceFns>(replaceFn)), ...);
    return replacer.replace(*this);
  }

  void
  walkSubElements(const llvm::function_ref<void(Type)> typeWalkFn,
                  const llvm::function_ref<void(Attribute)> attrWalkFn) const;
  Type replaceSubElements(llvm::ArrayRef<Type> typeReplaced,
                          llvm::ArrayRef<Attribute> attrReplaced) const;

  static Type getFromVoidPointer(void *impl) {
    return static_cast<TypeImpl *>(impl);
  }

  bool isSignedInt() const;

private:
  TypeImpl *impl;
};

class AbstractType {
public:
  using PrintFn = std::function<void(Type, llvm::raw_ostream &)>;
  using GetSizeAndAlignFn =
      std::function<std::pair<size_t, size_t>(Type, const StructSizeMap &)>;
  using WalkSubElementsFn =
      std::function<void(Type, llvm::function_ref<void(Type)>,
                         llvm::function_ref<void(Attribute)>)>;
  using ReplaceSubElementsFn = std::function<Type(Type, llvm::ArrayRef<Type>,
                                                  llvm::ArrayRef<Attribute>)>;

  TypeID getId() const { return id; }
  IRContext *getContext() const { return context; }
  const PrintFn &getPrintFn() const { return printFn; }
  const GetSizeAndAlignFn &getGetBitSizeFn() const { return getBitSizeFn; }
  const WalkSubElementsFn &getWalkSubElementsFn() const {
    return walkSubElementsFn;
  }
  const ReplaceSubElementsFn &getReplaceSubElementsFn() const {
    return replaceSubElementsFn;
  }

  template <typename T> static AbstractType build(IRContext *context) {
    TypeID typeId = TypeID::get<T>();
    return AbstractType(typeId, context, T::getPrintFn(),
                        T::getSizeAndAlignFn(), T::getWalkSubElementsFn(),
                        T::getReplaceSubElementsFn());
  }

private:
  AbstractType(TypeID id, IRContext *context, PrintFn printFn,
               GetSizeAndAlignFn getBitSizeFn,
               WalkSubElementsFn walkSubElementsFn,
               ReplaceSubElementsFn replaceSubElementsFn)
      : id(id), context(context), printFn(std::move(printFn)),
        getBitSizeFn(std::move(getBitSizeFn)),
        walkSubElementsFn(std::move(walkSubElementsFn)),
        replaceSubElementsFn(std::move(replaceSubElementsFn)) {}

  TypeID id;
  IRContext *context;
  PrintFn printFn;
  GetSizeAndAlignFn getBitSizeFn;
  WalkSubElementsFn walkSubElementsFn;
  ReplaceSubElementsFn replaceSubElementsFn;
};

class TypeImpl {
public:
  using ImplBase = TypeImpl;
  TypeImpl() = default;
  TypeID getId() const { return abstractType->getId(); }
  IRContext *getContext() const { return abstractType->getContext(); }

  const AbstractType::PrintFn &getPrintFn() const {
    return abstractType->getPrintFn();
  }

  const AbstractType::GetSizeAndAlignFn &getGetBitSizeFn() const {
    return abstractType->getGetBitSizeFn();
  }

  const AbstractType::WalkSubElementsFn &getWalkSubElementsFn() const {
    return abstractType->getWalkSubElementsFn();
  }

  const AbstractType::ReplaceSubElementsFn &getReplaceSubElementsFn() const {
    return abstractType->getReplaceSubElementsFn();
  }

  const AbstractType::GetSizeAndAlignFn &getSizeAndAlignFn() const {
    return abstractType->getGetBitSizeFn();
  }

private:
  friend class TypeStorage;
  void setAbstractTable(AbstractType *abstractType) {
    this->abstractType = abstractType;
  }
  AbstractType *abstractType;
};

struct TypeBuilder {
  template <typename T, typename... Args>
  static T get(IRContext *context, Args &&...args) {
    TypeStorage *storage = context->getTypeStorage();
    return storage->getImplByArgs<T>(std::forward<Args>(args)...);
  }

  template <typename T>
  static AbstractType *getAbstractType(IRContext *context) {
    TypeID typeId = TypeID::get<T>();
    TypeStorage *storage = context->getTypeStorage();
    return storage->getAbstractTable<AbstractType>(typeId);
  }
};

template <typename KeyType> struct TypeImplTemplate : public TypeImpl {
  using KeyTy = KeyType;

  bool isEqual(const KeyTy &otherKey) const { return key == otherKey; }

  const KeyTy &getKeyValue() const { return key; }

protected:
  TypeImplTemplate(KeyTy key) : key(key) {}

private:
  KeyTy key;
};

template <typename Range> static std::string typeRangeToString(Range &&range) {
  std::string result;
  llvm::raw_string_ostream os(result);
  for (Type item : range) {
    if (!result.empty()) {
      os << ", ";
    }
    item.print(os);
  }
  return result;
}

std::tuple<size_t, size_t, llvm::SmallVector<size_t>>
getTypeSizeAlignOffsets(llvm::ArrayRef<Type> fields,
                        const StructSizeMap &sizeMap);

} // namespace kecc::ir

namespace llvm {
template <> struct DenseMapInfo<kecc::ir::Type> {
  static kecc::ir::Type getEmptyKey() { return nullptr; }

  static kecc::ir::Type getTombstoneKey() {
    return DenseMapInfo<kecc::ir::TypeImpl *>::getTombstoneKey();
  }

  static unsigned getHashValue(kecc::ir::Type type) { return hash_value(type); }

  static bool isEqual(kecc::ir::Type lhs, kecc::ir::Type rhs) {
    return lhs == rhs;
  }
};

template <typename T>
struct DenseMapInfo<T, std::enable_if_t<std::is_base_of_v<kecc::ir::Type, T>>>
    : public DenseMapInfo<kecc::ir::Type> {
  static T getEmptyKey() { return nullptr; }

  static T getTombstoneKey() {
    return DenseMapInfo<kecc::ir::TypeImpl *>::getTombstoneKey();
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

template <typename T>
  requires(std::is_base_of_v<kecc::ir::Type, T>)
inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os, T type) {
  type.print(os);
  return os;
}

template <typename T>
  requires(std::is_base_of_v<kecc::ir::Type, T>)
inline std::ostream &operator<<(std::ostream &os, T type) {
  os << type.toString();
  return os;
}

#endif // KECC_IR_TYPES_H
