#ifndef KECC_IR_TYPE_ATTRIBUTE_SUPPORT_H
#define KECC_IR_TYPE_ATTRIBUTE_SUPPORT_H

#include "kecc/ir/Context.h"
#include "kecc/utils/MLIR.h"

namespace kecc::ir {

template <typename T>
concept HasVerify = requires(T) {
  { T::verify() } -> std::convertible_to<bool>;
};

template <typename ConcreteType, typename ParentType, typename BaseType,
          typename BuilderType, typename ImplType, typename AbstractTableType>
struct TypeAttrTemplate : public ParentType {
  using ImplTy = ImplType;
  using AbstractTableTy = AbstractTableType;
  using Base = TypeAttrTemplate<ConcreteType, ParentType, BaseType, BuilderType,
                                ImplTy, AbstractTableType>;
  using ParentType::ParentType;

  ImplTy *getImpl() const {
    return static_cast<ImplTy *>(ParentType::getImpl());
  }

  template <typename... Args>
  static ConcreteType get(IRContext *context, Args &&...args) {
    if constexpr (HasVerify<ConcreteType>) {
      if (!ConcreteType::verify(std::forward<Args>(args)...))
        return nullptr;
    }
    return BuilderType::template get<ConcreteType>(context,
                                                   std::forward<Args>(args)...);
  }

  template <typename T>
    requires(std::is_convertible_v<T, BaseType>)
  static bool classof(T type) {
    return type.getId() == TypeID::get<ConcreteType>();
  }

  static auto getPrintFn() {
    return [](BaseType obj, llvm::raw_ostream &os) {
      ConcreteType::printer(obj.template cast<ConcreteType>(), os);
    };
  }
};

} // namespace kecc::ir

#endif // KECC_IR_TYPE_ATTRIBUTE_SUPPORT_H
