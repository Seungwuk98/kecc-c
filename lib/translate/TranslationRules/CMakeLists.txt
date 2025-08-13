add_library(
  KeccTranslationRules
  STATIC  
  BinaryTranslation.cpp
  TranslateInstruction.cpp
)

target_link_libraries(KeccTranslationRules
  PRIVATE
  ${llvm_libs}
  MLIRSupport
  KeccIR
  KeccASM
)
