add_library(
  KeccDriver
  STATIC 
  Compilation.cpp
  Action.cpp
)

target_link_libraries(
  KeccDriver
  PRIVATE
  ${llvm_libs}
  KeccParser
  KeccIR 
  KeccIRAnalyses 
  KeccIRPasses
  KeccASM 
  KeccTranslate 
  KeccTranslatePasses 
  KeccC
)


