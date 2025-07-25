add_library(KeccASM
  STATIC 
  AsmInstruction.cpp
  Register.cpp
  Asm.cpp
)

target_link_libraries(KeccASM
  PRIVATE
  ${llvm_libs}
  KeccIR
  MLIRSupport
)
