
find_program(CLANG clang REQUIRED)
# TODO: require cross compile libraries
find_program(LLD lld REQUIRED)

find_program(QEMU_RISCV64_STATIC qemu-riscv64-static REQUIRED)

set(EXTRA_DEFINES "")
if (KECC_ENABLED_FUZZ_TEST)
  find_program(RISCV_GCC 
    riscv64-linux-gnu-gcc 
    riscv64-linux-gnu-gcc-13
    REQUIRED)
  message(STATUS "Found riscv64-linux-gnu-gcc: ${RISCV_GCC}")
  set(EXTRA_DEFINES "${EXTRA_DEFINES}\n#define RISCV64_GCC_DIR \"${RISCV_GCC}\"")
endif()

configure_file(DriverConfig.h.in ${CMAKE_CURRENT_BINARY_DIR}/DriverConfig.h)

