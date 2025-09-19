

add_library(
  KeccC 
  STATIC 
  ParseAST.cpp
  TypeConverter.cpp
  IRGenerator.cpp
)

target_link_libraries(
  KeccC 
  PRIVATE
  ${llvm_libs}
  clangTooling
)
