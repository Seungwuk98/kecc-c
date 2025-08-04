
add_library(KeccIRAnalyses
  STATIC 
  DominanceAnalysis.cpp
  VisitOrderAnalysis.cpp
  LiveRangeAnalysis.cpp
  LivenessAnalysis.cpp
  LoopAnalysis.cpp
)

target_link_libraries(KeccIRAnalyses
  PRIVATE
  KeccIR
  ${llvm_libs}
  MLIRSupport
)
