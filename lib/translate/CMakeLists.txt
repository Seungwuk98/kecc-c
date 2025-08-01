add_library(
  KeccTranslate
  STATIC 
  IRTranslater.cpp
)

target_link_libraries(KeccTranslate
  PRIVATE
  ${llvm_libs}
  MLIRSupport
  KeccIR
  KeccASM
)
