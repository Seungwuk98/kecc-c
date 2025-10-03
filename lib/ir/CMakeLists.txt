
add_library(KeccIR
  STATIC 
  IR.cpp
  Module.cpp
  Block.cpp
  Context.cpp
  Type.cpp
  TypeWalk.cpp
  IRTypes.cpp
  IRBuilder.cpp
  Instruction.cpp
  IRInstructions.cpp
  Attribute.cpp
  IRAttributes.cpp
  Value.cpp
  Pass.cpp
  JumpArg.cpp
  PatternMatch.cpp
  Interpreter.cpp
  InterpreterValue.cpp
)

add_subdirectory(Analyses)
add_subdirectory(Passes)

target_link_libraries(KeccIR
  PRIVATE
  ${llvm_libs}
  MLIRSupport
  KeccIRAnalyses
)
