cmake_minimum_required(VERSION 3.28)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
project("kecc-c")

find_package(LLVM REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)

llvm_map_components_to_libnames(llvm_libs support core irreader)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(include)


add_subdirectory(lib)
add_subdirectory(tools)
