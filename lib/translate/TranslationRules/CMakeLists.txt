add_library(
  KeccTranslationRules
  STATIC  
  BinaryTranslation.cpp
  UnaryTranslation.cpp
  ConstantTranslation.cpp
  MemoryInstTranslation.cpp
  TranslateInstruction.cpp
  TypeCastTranslation.cpp
)

target_link_libraries(KeccTranslationRules
  PRIVATE
  ${llvm_libs}
  MLIRSupport
  KeccIR
  KeccASM
)
