cmake_minimum_required(VERSION 3.28)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
project("kecc-c")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(AddKeccDriver)

find_package(LLVM REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)


llvm_map_components_to_libnames(llvm_libs support core irreader)

find_package(MLIR REQUIRED CONFIG)
find_package(Clang REQUIRED CONFIG)

if (KECC_USE_MOLD)
  find_program(MOLD mold REQUIRED)

  set(CMAKE_EXE_LINKER_FLAGS 
    ${CMAKE_EXE_LINKER_FLAGS} "-fuse-ld=mold"
  )
  set(CMAKE_SHARED_LINKER_FLAGS 
    ${CMAKE_SHARED_LINKER_FLAGS} 
    "-fuse-ld=mold"
  )
  message(STATUS "Using mold as the linker")      
endif()

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CLANG_INCLUDE_DIRS})
include_directories(include)
include_directories(${CMAKE_BINARY_DIR}/include)
include_directories(third-party/doctest)

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(third-party)
add_subdirectory(test)

