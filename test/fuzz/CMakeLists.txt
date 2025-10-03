
find_program(CSMITH_PATH csmith REQUIRED)
message(STATUS "CSmith found at: ${CSMITH_PATH}")

find_program(GCC_PATH gcc REQUIRED)
message(STATUS "GCC found at: ${GCC_PATH}")

find_program(CREDUCE_PATH creduce REQUIRED)
message(STATUS "CReduce found at: ${CREDUCE_PATH}")

find_file(
  CSMITH_HEADER 
  csmith/csmith.h 
  PATHS /usr/include/ /usr/local/include/
)

if(NOT CSMITH_HEADER)
  message(FATAL_ERROR "csmith.h not found. Please install CSmith and ensure it is in your include path.")
else()
  message(STATUS "csmith.h found at: ${CSMITH_HEADER}")
endif()

get_filename_component(CSMITH_INCLUDE_DIR ${CSMITH_HEADER} DIRECTORY)
message(STATUS "CSmith include directory: ${CSMITH_INCLUDE_DIR}")


configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/FuzzConfig.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/FuzzConfig.h"
)

include_directories("${CMAKE_CURRENT_BINARY_DIR}")

add_executable(
  fuzz-test
  fuzz-test.cpp
)

target_link_libraries(
  fuzz-test 
  PRIVATE 
  ${llvm_libs}
  KeccDriver
)

add_dependencies(fuzz-test kecc fuzz keci)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/succeed_fuzz_scripts.sh.in 
  ${CMAKE_CURRENT_BINARY_DIR}/succeed_fuzz_scripts.sh
)
