#ifndef KECC_IR_TYPE_WALK_H
#define KECC_IR_TYPE_WALK_H

#include "kecc/ir/TypeConcepts.h"
#include "kecc/ir/WalkSupport.h"
#include "kecc/utils/LogicalResult.h"
#include "kecc/utils/TypeTraits.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include <type_traits>
#include <utility>

namespace kecc::ir {

class Type;
class Attribute;

namespace detail {

using TypeWalkFn = llvm::function_ref<void(Type)>;
using AttrWalkFn = llvm::function_ref<void(Attribute)>;

class WalkFnApplier {
public:
  WalkFnApplier(TypeWalkFn typeWalkFn, AttrWalkFn attrWalkFn)
      : typeWalkFn(std::move(typeWalkFn)), attrWalkFn(std::move(attrWalkFn)) {}

  template <typename T>
    requires std::is_base_of_v<Type, T>
  void walk(T type) const {
    typeWalkFn(type);
  }

  template <typename T>
    requires std::is_base_of_v<Attribute, T>
  void walk(T attr) const {
    attrWalkFn(attr);
  }

private:
  TypeWalkFn typeWalkFn;
  AttrWalkFn attrWalkFn;
};

template <typename BaseT> class ReplacedAdvancer {
public:
  ReplacedAdvancer(llvm::ArrayRef<BaseT> replaced) : replaced(replaced) {}
  BaseT advance() {
    assert(!replaced.empty() && "No replacements available");
    BaseT current = replaced.front();
    replaced = replaced.drop_front();
    return current;
  }

private:
  llvm::ArrayRef<BaseT> replaced;
};

template <typename T, typename Enable = void> struct TypeAttrWalker {
  static void walk(T, const WalkFnApplier &) {}

  template <typename Ty, typename At>
  static auto replace(T obj, ReplacedAdvancer<Ty> &, ReplacedAdvancer<At> &) {
    return std::move(obj);
  }
};

template <typename T>
struct TypeAttrWalker<T, std::enable_if_t<std::is_base_of_v<Type, T>>> {
  static void walk(const T &type, const WalkFnApplier &applier) {
    applier.walk(type);
  }

  template <typename Ty, typename At>
  static auto replace(T type, ReplacedAdvancer<Ty> &typeAdvancer,
                      ReplacedAdvancer<At> & /*attrAdvancer*/) {
    return typeAdvancer.advance().template cast<T>();
  }
};

template <typename T>
struct TypeAttrWalker<T, std::enable_if_t<std::is_base_of_v<Attribute, T>>> {
  static void walk(T attr, const WalkFnApplier &walkFn) { walkFn.walk(attr); }

  template <typename Ty, typename At>
  static auto replace(T type, ReplacedAdvancer<Ty> & /*typeAdvancer*/,
                      ReplacedAdvancer<At> &attrAdvancer) {
    return attrAdvancer.advance().template cast<T>();
  }
};

template <typename F, typename S> struct TypeAttrWalker<std::pair<F, S>> {
  static void walk(const std::pair<F, S> &obj, const WalkFnApplier &applier) {
    TypeAttrWalker<std::remove_cvref_t<F>>::walk(obj.first, applier);
    TypeAttrWalker<std::remove_cvref_t<S>>::walk(obj.second, applier);
  }

  template <typename Ty, typename At>
  static auto replace(const std::pair<F, S> &obj,
                      ReplacedAdvancer<Ty> &typeAdvancer,
                      ReplacedAdvancer<At> &attrAdvancer) {
    auto first = TypeAttrWalker<std::remove_cvref_t<F>>::replace(
        obj.first, typeAdvancer, attrAdvancer);
    auto second = TypeAttrWalker<std::remove_cvref_t<S>>::replace(
        obj.second, typeAdvancer, attrAdvancer);
    return std::make_pair(first, second);
  }
};

template <typename... Ts> struct TypeAttrWalker<std::tuple<Ts...>> {
  static void walk(const std::tuple<Ts...> &type,
                   const WalkFnApplier &applier) {
    std::apply(
        [&]<typename... Args>(Args &&...args) {
          (TypeAttrWalker<std::remove_cvref_t<Args>>::walk(
               std::forward<Args>(args), applier),
           ...);
        },
        type);
  }

  template <typename Ty, typename At>
  static auto replace(const std::tuple<Ts...> &type,
                      ReplacedAdvancer<Ty> &typeAdvancer,
                      ReplacedAdvancer<At> &attrAdvancer) {
    return std::apply(
        [&]<typename... Args>(Args &&...args) {
          return std::make_tuple(
              TypeAttrWalker<std::remove_cvref_t<Args>>::replace(
                  std::forward<Args>(args), typeAdvancer, attrAdvancer)...);
        },
        type);
  }
};

template <typename T> struct TypeAttrWalker<llvm::ArrayRef<T>> {
  static void walk(llvm::ArrayRef<T> type, const WalkFnApplier &walkFn) {
    for (const auto &elem : type) {
      TypeAttrWalker<std::remove_cvref_t<T>>::walk(elem, walkFn);
    }
  }

  template <typename Ty, typename At>
  static auto replace(const llvm::ArrayRef<T> &type,
                      ReplacedAdvancer<Ty> &typeAdvancer,
                      ReplacedAdvancer<At> &attrAdvancer) {
    llvm::SmallVector<T> result;
    result.reserve(type.size());
    for (const auto &elem : type) {
      result.push_back(TypeAttrWalker<std::remove_cvref_t<T>>::replace(
          elem, typeAdvancer, attrAdvancer));
    }
    return result;
  }
};

template <typename ConcreteT, typename WalkTypeFn, typename WalkAttrFn>
void walkSubElementImpl(ConcreteT type, WalkTypeFn &&typeFn,
                        WalkAttrFn &&attrFn) {
  using ImplTy = typename ConcreteT::ImplTy;
  if constexpr (DetectGetKeyValue<ImplTy>) {
    using KeyTy = typename ImplTy::KeyTy;
    const KeyTy &key = type.getImpl()->getKeyValue();

    WalkFnApplier applier(std::forward<WalkTypeFn>(typeFn),
                          std::forward<WalkAttrFn>(attrFn));
    detail::TypeAttrWalker<std::remove_cvref_t<KeyTy>>::walk(key, applier);
  }
}

template <typename ConcreteT>
auto replaceSubElementImpl(ConcreteT type, llvm::ArrayRef<Type> replacedType,
                           llvm::ArrayRef<Attribute> replacedAttr) {
  using ImplTy = typename ConcreteT::ImplTy;
  if constexpr (DetectGetKeyValue<ImplTy>) {
    using KeyTy = typename ImplTy::KeyTy;
    const KeyTy &key = type.getImpl()->getKeyValue();

    ReplacedAdvancer<Type> typeAdvancer(replacedType);
    ReplacedAdvancer<Attribute> attrAdvancer(replacedAttr);
    auto newKey =
        detail::TypeAttrWalker<KeyTy>::replace(key, typeAdvancer, attrAdvancer);
    using NewKeyTy = std::remove_cvref_t<decltype(newKey)>;
    if constexpr (utils::IsTupleV<NewKeyTy> || utils::IsPairV<NewKeyTy>) {
      return std::apply(
          [&]<typename... Args>(Args &&...args) {
            return ConcreteT::get(type.getContext(),
                                  std::forward<Args>(args)...);
          },
          std::move(newKey));
    } else {
      return ConcreteT::get(type.getContext(), std::move(newKey));
    }
  }
  // no replacement
  return type;
}

} // namespace detail

enum class TypeWalkOrder {
  PreOrder,
  PostOrder,
};

class TypeWalker {
public:
  enum Order {
    PreOrder,
    PostOrder,
  };

  TypeWalker(Order order = PreOrder) : order(order) {}

  WalkResult walk(Type type) const;
  WalkResult walk(Attribute attr) const;

  template <typename Fn> void addWalkFn(Fn &&walkFn) {
    addWalkFnImpl(std::forward<Fn>(walkFn));
  }

private:
  void addWalkFnImpl(llvm::function_ref<WalkResult(Type)> typeWalkFn) {
    typeWalkFns.emplace_back(std::move(typeWalkFn));
  }
  void addWalkFnImpl(llvm::function_ref<WalkResult(Attribute)> attrWalkFn) {
    attrWalkFns.emplace_back(std::move(attrWalkFn));
  }

  friend class TypeWalkerDetail;
  Order order;
  llvm::SmallVector<llvm::function_ref<WalkResult(Type)>> typeWalkFns;
  llvm::SmallVector<llvm::function_ref<WalkResult(Attribute)>> attrWalkFns;
};

template <typename T> using ReplaceResult = std::pair<T, utils::LogicalResult>;

class TypeReplacer {
public:
  TypeReplacer();
  ~TypeReplacer();

  Type replace(Type type);
  Attribute replace(Attribute attr);

  template <typename... ReplaceFns>
  void addReplaceFn(ReplaceFns &&...replaceFn) {
    (addReplaceFnImpl(std::forward<ReplaceFns>(replaceFn)), ...);
  }

private:
  friend class TypeReplacerDetail;

  void
  addReplaceFnImpl(llvm::function_ref<ReplaceResult<Type>(Type)> replaceTypeFn);
  void addReplaceFnImpl(
      llvm::function_ref<ReplaceResult<Attribute>(Attribute)> replaceAttrFn);

  llvm::DenseMap<void *, void *>
      visited; // void ptr type is for avoid cyclic dependencies
  llvm::SmallVector<llvm::function_ref<ReplaceResult<Type>(Type)>>
      replaceTypeFn;
  llvm::SmallVector<llvm::function_ref<ReplaceResult<Attribute>(Attribute)>>
      replaceAttrFn;
};

} // namespace kecc::ir

#endif // KECC_IR_TYPE_WALK_H
