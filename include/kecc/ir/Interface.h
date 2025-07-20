#ifndef KECC_IR_INTERFACE_H
#define KECC_IR_INTERFACE_H

#include "kecc/ir/Context.h"
#include "kecc/utils/MLIR.h"
#include "llvm/ADT/DenseMap.h"

namespace kecc::ir {

template <typename ConcreteType, typename BaseType, typename Traits>
class Interface : public BaseType {
public:
  using Concept = Traits::Concept;
  struct Trait {

    using Model = typename Traits::template Model<ConcreteType>;

    static TypeID getInterfaceId() { return TypeID::get<ConcreteType>(); }
  };

  Interface(std::nullptr_t) : BaseType(nullptr), impl(nullptr) {}

  Interface(BaseType obj)
      : BaseType(obj),
        impl(ConcreteType::template getInterfaceFor<ConcreteType>(obj)) {}

  static bool classof(BaseType type) { return ConcreteType::classof(type); }

private:
  const Concept *impl;
};

template <typename T>
concept HasGetInterfaceID = requires {
  { T::Trait::getInterfaceId() } -> std::convertible_to<TypeID>;
};

class InterfaceMap {
public:
  InterfaceMap() = default;

  template <typename... Trait> static InterfaceMap build(IRContext *context) {
    InterfaceMap map;
    (map.buildInterfaceAndInsert<Trait>(context), ...);
    return map;
  }

  template <typename Interface> typename Interface::Model *lookup() const {
    TypeID typeId = Interface::Trait::getInterfaceId();
    auto it = interfaces.find(typeId);
    if (it != interfaces.end()) {
      return static_cast<typename Interface::Model *>(it->second);
    }
    return nullptr;
  }

  template <typename Interface> bool contains() const {
    return contains(Interface::Trait::getInterfaceId());
  }

  bool contains(TypeID typeId) const { return interfaces.contains(typeId); }

private:
  template <typename Trait> void buildInterfaceAndInsert(IRContext *context) {
    if constexpr (HasGetInterfaceID<Trait>) {
      TypeID typeId = Trait::getInterfaceId();
      if (!interfaces.contains(typeId)) {
        auto *model = new (context->allocate(sizeof(typename Trait::Model)))
            typename Trait::Model();
        interfaces.try_emplace(typeId, model);
      }
    }
  }

  llvm::DenseMap<TypeID, void *> interfaces;
};

} // namespace kecc::ir

#endif // KECC_IR_INTERFACE_H
