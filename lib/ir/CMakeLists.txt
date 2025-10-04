
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

if (KECC_ENABLED_FUZZ_TEST) 
  # Some cases, clang optimization causes transformation when calculating float values or casting
  # which makes fuzz testing fail.
  message(STATUS "Fuzz testing enabled: Disabling optimizations for InterpreterValue.cpp")
  set_source_files_properties(
    InterpreterValue.cpp
    PROPERTIES COMPILE_FLAGS "-O0" 
  )
endif()

add_subdirectory(Analyses)
add_subdirectory(Passes)

target_link_libraries(KeccIR
  PRIVATE
  ${llvm_libs}
  MLIRSupport
  KeccIRAnalyses
)

