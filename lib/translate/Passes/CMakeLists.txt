

add_library(KeccTranslatePasses
  STATIC 
  ConversionToCopy.cpp
  InlineMemoryInst.cpp
)

target_link_libraries(KeccTranslatePasses
  PRIVATE
  KeccTranslate
  ${llvm_libs}
  MLIRSupport
)
