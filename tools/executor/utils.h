#ifndef KECC_EXECUTOR_UTILS_H
#define KECC_EXECUTOR_UTILS_H

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

inline bool random_bool() {
  int r = random();
  return r & 1;
}

inline int8_t random_i8() { return (int8_t)(random()); }
inline int16_t random_i16() { return (int16_t)(random()); }
inline int32_t random_i32() { return (int32_t)(random()); }
inline int64_t random_i64() {
  int upper = (int32_t)(random());
  int lower = (int32_t)(random());
  return ((int64_t)upper << 32) | (uint32_t)lower;
}

inline uint8_t random_u8() { return (uint8_t)(random()); }
inline uint16_t random_u16() { return (uint16_t)(random()); }
inline uint32_t random_u32() { return (uint32_t)(random()); }
inline uint64_t random_u64() {
  uint32_t upper = (uint32_t)(random());
  uint32_t lower = (uint32_t)(random());
  return ((uint64_t)upper << 32) | (uint64_t)lower;
}

#endif // KECC_EXECUTOR_UTILS_H
