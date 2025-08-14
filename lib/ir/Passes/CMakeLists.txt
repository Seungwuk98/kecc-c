
add_library(KeccIRPasses
  STATIC 
  CanonicalizeConstantPass.cpp
  CanonicalizeStruct.cpp
  SimplifyCFG.cpp
  Mem2Reg.cpp
  GVN.cpp
  DeadCode.cpp
  OutlineConstant.cpp
  InstructionFold.cpp
  OutlineMultipleResults.cpp
  InlineCall.cpp
  CreateFunctionArgument.cpp
)

target_link_libraries(KeccIRPasses
  PRIVATE
  KeccIR
  KeccIRAnalyses
  ${llvm_libs}
  MLIRSupport
)
