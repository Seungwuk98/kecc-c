#ifndef KECC_TRANSLATE_CONSTANTS_H
#define KECC_TRANSLATE_CONSTANTS_H

#include <cstdint>
namespace kecc {

constexpr std::int64_t MAX_INT_12 = (1 << 11) - 1;
constexpr std::int64_t MIN_INT_12 = -(1 << 11);

} // namespace kecc

#endif // KECC_TRANSLATE_CONSTANTS_H
