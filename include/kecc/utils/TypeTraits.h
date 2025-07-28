#ifndef KECC_UTILS_STL_H
#define KECC_UTILS_STL_H

#include <tuple>

namespace kecc::utils {

template <typename T> struct IsTuple {
  static constexpr bool value = false;
};

template <typename... Ts> struct IsTuple<std::tuple<Ts...>> {
  static constexpr bool value = true;
};

template <typename T> bool IsTupleV = IsTuple<T>::value;

template <typename T> struct IsPair {
  static constexpr bool value = false;
};

template <typename F, typename S> struct IsPair<std::pair<F, S>> {
  static constexpr bool value = true;
};

template <typename T> bool IsPairV = IsPair<T>::value;

} // namespace kecc::utils

#endif // KECC_UTILS_STL_H
