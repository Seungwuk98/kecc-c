
find_program(CLANG clang REQUIRED)
# TODO: require cross compile libraries
find_program(LLD lld REQUIRED)

find_program(QEMU_RISCV64_STATIC qemu-riscv64-static REQUIRED)

configure_file(DriverConfig.h.in ${CMAKE_CURRENT_BINARY_DIR}/DriverConfig.h)

