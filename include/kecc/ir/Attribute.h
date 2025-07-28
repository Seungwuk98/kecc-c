#ifndef KECC_IR_ATTRIBUTE_H
#define KECC_IR_ATTRIBUTE_H

#include "kecc/ir/Context.h"
#include "kecc/ir/TypeAttributeSupport.h"
#include "kecc/utils/MLIR.h"
#include <string>

namespace kecc::ir {

class AttributeImpl;
struct AttributeBuilder;
class AbstractAttribute;

class Attribute {
public:
  template <typename ConcreteAttr, typename ParentAttr, typename ImplType>
  using Base = TypeAttrTemplate<ConcreteAttr, ParentAttr, Attribute,
                                AttributeBuilder, ImplType, AbstractAttribute>;

  Attribute() : impl(nullptr) {}
  Attribute(const Attribute &attr) : impl(attr.impl) {}
  Attribute(AttributeImpl *impl) : impl(impl) {}

  template <typename... U> bool isa() const { return llvm::isa<U...>(*this); }
  template <typename U> U cast() const { return llvm::cast<U>(*this); }
  template <typename U> U dyn_cast() const { return llvm::dyn_cast<U>(*this); }
  template <typename U> U dyn_cast_or_null() const {
    return llvm::dyn_cast_or_null<U>(*this);
  }

  operator bool() const { return impl != nullptr; }

  bool operator==(const Attribute &other) const { return impl == other.impl; }
  bool operator!=(const Attribute &other) const { return impl != other.impl; }

  AttributeImpl *getImpl() const { return impl; }

  IRContext *getContext() const;

  TypeID getId() const;

  friend inline llvm::hash_code hash_value(const Attribute &attr) {
    return llvm::DenseMapInfo<AttributeImpl *>::getHashValue(attr.impl);
  }

private:
  AttributeImpl *impl;
};

class AbstractAttribute {
public:
  TypeID getId() const { return id; }
  IRContext *getContext() const { return context; }

  template <typename T> static AbstractAttribute build(IRContext *context) {
    TypeID typeId = TypeID::get<T>();
    return AbstractAttribute(typeId, context);
  }

private:
  AbstractAttribute(TypeID id, IRContext *context) : id(id), context(context) {}
  TypeID id;
  IRContext *context;
};

class AttributeImpl {
public:
  using ImplBase = AttributeImpl;

  AttributeImpl() = default;
  TypeID getId() const { return abstractAttr->getId(); }
  IRContext *getContext() const { return abstractAttr->getContext(); }
  AbstractAttribute *getAbstractAttribute() const { return abstractAttr; }

private:
  friend class TypeStorage;
  void setAbstractTable(AbstractAttribute *attr) { abstractAttr = attr; }
  AbstractAttribute *abstractAttr;
};

struct AttributeBuilder {
  template <typename ConcreteAttr, typename... Args>
  static ConcreteAttr get(IRContext *context, Args &&...args) {
    TypeStorage *storage = context->getTypeStorage();
    return storage->getImplByArgs<ConcreteAttr>(std::forward<Args>(args)...);
  }

  template <typename T>
  static AbstractAttribute *getAbstractAttribute(IRContext *context) {
    TypeID typeId = TypeID::get<T>();
    TypeStorage *storage = context->getTypeStorage();
    return storage->getAbstractTable<AbstractAttribute>(typeId);
  }
};

template <typename KeyType> class AttributeImplTemplate : public AttributeImpl {
public:
  using KeyTy = KeyType;

  bool isEqual(const KeyTy &otherKey) const { return key == otherKey; }

  const KeyTy &getKeyValue() const { return key; }

protected:
  AttributeImplTemplate(const KeyType &key) : key(key) {}

private:
  KeyType key;
};

} // namespace kecc::ir

namespace llvm {

template <> struct DenseMapInfo<kecc::ir::Attribute> {
  static kecc::ir::Attribute getEmptyKey() { return nullptr; }

  static kecc::ir::Attribute getTombstoneKey() {
    return DenseMapInfo<kecc::ir::AttributeImpl *>::getTombstoneKey();
  }

  static unsigned getHashValue(kecc::ir::Attribute attr) {
    return hash_value(attr);
  }

  static bool isEqual(kecc::ir::Attribute lhs, kecc::ir::Attribute rhs) {
    return lhs == rhs;
  }
};

template <typename T>
struct DenseMapInfo<T,
                    std::enable_if_t<std::is_base_of_v<kecc::ir::Attribute, T>>>
    : public DenseMapInfo<kecc::ir::Attribute> {
  static T getEmptyKey() { return nullptr; }

  static T getTombstoneKey() {
    return DenseMapInfo<kecc::ir::AttributeImpl *>::getTombstoneKey();
  }
};

template <typename To, typename From>
struct CastInfo<To, From,
                std::enable_if_t<std::is_same_v<kecc::ir::Attribute,
                                                std::remove_const_t<From>> ||
                                 std::is_base_of_v<kecc::ir::Attribute, From>>>
    : NullableValueCastFailed<To>,
      DefaultDoCastIfPossible<To, From, CastInfo<To, From>> {
  static inline bool isPossible(kecc::ir::Attribute ty) {
    if constexpr (std::is_base_of_v<To, From>) {
      return true;
    } else {
      return To::classof(ty);
    };
  }
  static inline To doCast(kecc::ir::Attribute ty) { return To(ty.getImpl()); }
};

} // namespace llvm

template <typename T>
  requires(std::is_convertible_v<T, kecc::ir::Attribute>)
inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os, T attr) {
  attr.print(os);
  return os;
}

template <typename T>
  requires(std::is_convertible_v<T, kecc::ir::Attribute>)
inline std::ostream &operator<<(std::ostream &os, T attr) {
  os << attr.toString();
  return os;
}

#endif // KECC_IR_ATTRIBUTE_H
