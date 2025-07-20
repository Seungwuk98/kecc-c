
add_library(KeccIR
  STATIC 
  IR.cpp
  Module.cpp
  Block.cpp
  Context.cpp
  Type.cpp
  IRTypes.cpp
  IRBuilder.cpp
  Instruction.cpp
  IRInstructions.cpp
  Attribute.cpp
  IRAttributes.cpp
  Value.cpp
  Pass.cpp
  JumpArg.cpp
)

target_link_libraries(KeccIR
  PRIVATE
  ${llvm_libs}
  MLIRSupport
)

add_subdirectory(Analyses)
add_subdirectory(Passes)
