#ifndef KECC_UTILS_TYPEID_H
#define KECC_UTILS_TYPEID_H

#include <cstdint>
namespace kecc::utils {

using TypeId = intptr_t;

class SelfOwningId {
public:
  SelfOwningId() = default;
  SelfOwningId(const SelfOwningId &) = delete;
  SelfOwningId &operator=(const SelfOwningId &) = delete;
  TypeId getId() const { return reinterpret_cast<TypeId>(this); }
};

namespace detail {
template <typename T> struct TypeIdStorage {};
} // namespace detail

template <typename T> TypeId getId = detail::TypeIdStorage<T>::getId();

} // namespace kecc::utils

#define DECLARE_KECC_TYPE_ID(T)                                                \
  namespace kecc::utils::detail {                                              \
  template <> class TypeIdStorage<T> {                                         \
  public:                                                                      \
    static TypeId getId() { return id.getId(); }                               \
                                                                               \
  private:                                                                     \
    static SelfOwningId id;                                                    \
  };                                                                           \
  }

#define DEFINE_KECC_TYPE_ID(T)                                                 \
  namespace kecc::utils::detail {                                              \
  SelfOwningId TypeIdStorage<T>::id;                                           \
  }

#endif // KECC_UTILS_TYPEID_H
