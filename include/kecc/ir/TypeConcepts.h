#ifndef KECC_IR_TYPE_CONCEPTS_H
#define KECC_IR_TYPE_CONCEPTS_H

#include <utility>

namespace kecc::ir {
template <typename T, typename... Args>
concept DetectGetKey = requires(T obj, Args &&...args) {
  {
    T::getKey(std::forward<Args>(args)...)
  } -> std::convertible_to<typename T::KeyTy>;
};

template <typename T>
concept DetectGetKeyValue = requires(T *obj) {
  { obj->getKeyValue() } -> std::convertible_to<typename T::KeyTy>;
};

} // namespace kecc::ir

#endif // KECC_IR_TYPE_CONCEPTS_H
