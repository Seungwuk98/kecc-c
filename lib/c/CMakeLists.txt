

add_library(
  KeccC 
  STATIC 
  ParseAST.cpp
)

target_link_libraries(
  KeccC 
  PRIVATE
  ${llvm_libs}
  clangTooling
)
