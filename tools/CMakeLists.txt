add_executable(kecc-opt kecc-opt.cpp)

target_link_libraries(
  kecc-opt
  PRIVATE
  ${llvm_libs}  
  KeccParser
  KeccIRAnalyses
  KeccIRPasses
  KeccTranslate
)

add_executable(kecc-translate kecc-translate.cpp)

target_link_libraries(
  kecc-translate
  PRIVATE
  ${llvm_libs}
  KeccIR
  KeccParser
  KeccIRAnalyses
  KeccIRPasses
  KeccASM
  KeccTranslate
)

add_executable(kecc-test-executor kecc-test-executor.cpp)

find_program(CLANG clang REQUIRED)
# TODO: require cross compile libraries
find_program(LLD lld REQUIRED)

find_program(QEMU_RISCV64_STATIC qemu-riscv64-static REQUIRED)

set(KECC_TRANSLATE_DIR ${CMAKE_CURRENT_BINARY_DIR}/kecc-translate)

configure_file(ExecutorConfig.h.in ${CMAKE_CURRENT_BINARY_DIR}/ExecutorConfig.h)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

target_link_libraries(
  kecc-test-executor
  PRIVATE
  ${llvm_libs}
  KeccIR
  KeccParser
  KeccIRAnalyses
  KeccIRPasses
  KeccASM
  KeccTranslate
)

