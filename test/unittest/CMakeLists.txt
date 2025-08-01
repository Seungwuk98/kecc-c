
# get .cpp files in ir directory

file(GLOB_RECURSE IR_UNIT_TESTS ${CMAKE_CURRENT_SOURCE_DIR}/ir "*.cpp")
file(GLOB_RECURSE ASM_UNIT_TESTS ${CMAKE_CURRENT_SOURCE_DIR}/ir "*.cpp")

add_executable(
  UnitTest
  unittest.cpp
  ${IR_UNIT_TESTS}
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(ir)
add_subdirectory(asm)

target_link_libraries(
  UnitTest
  PUBLIC
  ${llvm_libs}
  doctest
  KeccIR
  KeccParser
  KeccASM
)
