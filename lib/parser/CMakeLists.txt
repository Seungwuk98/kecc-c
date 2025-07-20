add_library(KeccParser
  STATIC 
  Parser.cpp
  Lexer.cpp
)

target_link_libraries(
  KeccParser
  PRIVATE
  ${llvm_libs}
  KeccIR
)
