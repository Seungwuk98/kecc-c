
add_library(
  KeccTranslate
  STATIC 
  IRTranslater.cpp
  InterferenceGraph.cpp
  SpillAnalysis.cpp
  TranslateContext.cpp
  LiveRangeAnalysis.cpp
  LivenessAnalysis.cpp
  CallLivenessAnalysis.cpp
  RegisterAllocation.cpp
  FunctionStack.cpp
  DefaultTranslationRules.cpp
  IntermediateInsts.cpp
)

target_link_libraries(KeccTranslate
  PRIVATE
  ${llvm_libs}
  MLIRSupport
  KeccIR
  KeccASM
)

add_subdirectory(Passes)
