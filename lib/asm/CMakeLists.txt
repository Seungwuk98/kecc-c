add_library(KeccASM
  STATIC 
  ASMInstruction.cpp
)

target_link_libraries(KeccASM
  PRIVATE
  ${llvm_libs}
  MLIRSupport
)
