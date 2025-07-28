#ifndef KECC_IR_TYPE_WALK_H
#define KECC_IR_TYPE_WALK_H

#include "kecc/ir/TypeConcepts.h"
#include "kecc/ir/WalkSupport.h"
#include "kecc/utils/TypeTraits.h"
#include "llvm/ADT/ArrayRef.h"
#include <type_traits>
#include <utility>

namespace kecc::ir {

class Type;
namespace detail {

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

template <typename T, typename Enable = void> struct TypeWalker {
  template <typename WalkFn> static void walk(T type, WalkFn &&walkFn) {}

  template <typename BaseT>
  static auto replace(T obj, ReplacedAdvancer<BaseT> &advancer) {
    return std::move(obj);
  }
};

template <typename T>
struct TypeWalker<T, std::enable_if_t<std::is_base_of_v<Type, T>>> {
  template <typename WalkFn> static void walk(const T &type, WalkFn &&walkFn) {
    std::forward<WalkFn>(walkFn)(type);
  }

  template <typename BaseT>
  static auto replace(T type, ReplacedAdvancer<BaseT> &replaced) {
    return replaced.advance();
  }
};

template <typename F, typename S> struct TypeWalker<std::pair<F, S>> {
  template <typename WalkFn>
  static void walk(const std::pair<F, S> &type, WalkFn &&walkFn) {
    TypeWalker<std::remove_cvref_t<F>>::walk(type.first,
                                             std::forward<WalkFn>(walkFn));
    TypeWalker<std::remove_cvref_t<S>>::walk(type.second,
                                             std::forward<WalkFn>(walkFn));
  }

  template <typename BaseT>
  static auto replace(std::pair<F, S> type, ReplacedAdvancer<BaseT> &replaced) {
    auto first = TypeWalker<std::remove_cvref_t<F>>::replace(replaced);
    auto second = TypeWalker<std::remove_cvref_t<S>>::replace(replaced);
    return std::make_pair(first, second);
  }
};

template <typename... Ts> struct TypeWalker<std::tuple<Ts...>> {
  template <typename WalkFn>
  static void walk(const std::tuple<Ts...> &type, WalkFn &&walkFn) {
    std::apply(
        [&]<typename... Args>(Args &&...args) {
          (TypeWalker<std::remove_cvref_t<Args>>::walk(
               std::forward<Args>(args), std::forward<WalkFn>(walkFn)),
           ...);
        },
        type);
  }

  template <typename BaseT>
  static auto replace(const std::tuple<Ts...> &type,
                      ReplacedAdvancer<BaseT> &replaced) {
    return std::apply(
        [&]<typename... Args>(Args &&...args) {
          return std::make_tuple(
              TypeWalker<std::remove_cvref_t<Args>>::replace(replaced)...);
        },
        type);
  }
};

template <typename T> struct TypeWalker<llvm::ArrayRef<T>> {
  template <typename WalkFn>
  static void walk(const llvm::ArrayRef<T> &type, WalkFn &&walkFn) {
    for (const auto &elem : type) {
      TypeWalker<std::remove_cvref_t<T>>::walk(elem,
                                               std::forward<WalkFn>(walkFn));
    }
  }

  template <typename BaseT>
  static llvm::SmallVector<BaseT> replace(const llvm::ArrayRef<T> &type,
                                          ReplacedAdvancer<BaseT> &replaced) {
    llvm::SmallVector<BaseT> result;
    result.reserve(type.size());
    for (const auto &elem : type) {
      result.push_back(
          TypeWalker<std::remove_cvref_t<T>>::replace(elem, replaced));
    }
    return result;
  }
};

} // namespace detail

template <typename ConcreteT, typename WalkFn>
void walkSubElementImpl(ConcreteT type, WalkFn &&fn) {
  using ImplTy = typename ConcreteT::ImplTy;
  if constexpr (DetectKeyTy<ImplTy>) {
    using KeyTy = typename ImplTy::KeyTy;
    const KeyTy &key = type.getImpl()->getKeyValue();

    detail::TypeWalker<std::remove_cvref_t<ConcreteT>>::walk(
        key, std::forward<WalkFn>(fn));
  }
}

template <typename ConcreteT>
auto replaceSubElementImpl(ConcreteT type, llvm::ArrayRef<Type> replaced) {
  using ImplTy = typename ConcreteT::ImplTy;
  if constexpr (DetectKeyTy<ImplTy>) {
    using KeyTy = typename ImplTy::KeyTy;
    const KeyTy &key = type.getImpl()->getKeyValue();

    detail::ReplacedAdvancer<Type> advancer(replaced);
    auto newKey = detail::TypeWalker<KeyTy>::replace(key, advancer);
    using NewKeyTy = std::remove_cvref_t<decltype(newKey)>;
    if constexpr (utils::IsTupleV<NewKeyTy>) {
      return std::apply(
          [&]<typename... Args>(Args &&...args) {
            return ConcreteT::get(type.getContext(),
                                  std::forward<Args>(args)...);
          },
          std::move(newKey));
    } else if constexpr (utils::IsPairV<NewKeyTy>) {
      return ConcreteT::get(type.getContext(), newKey.first, newKey.second);
    } else {
      return ConcreteT::get(type.getContext(), std::move(newKey));
    }
  }
  // no replacement
  return type;
}

} // namespace kecc::ir

#endif // KECC_IR_TYPE_WALK_H
