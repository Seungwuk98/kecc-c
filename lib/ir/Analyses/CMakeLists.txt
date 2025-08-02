
add_library(KeccIRAnalyses
  STATIC 
  DominanceAnalysis.cpp
  VisitOrderAnalysis.cpp
  LiveRangeAnalysis.cpp
  LivenessAnalysis.cpp
)

target_link_libraries(KeccIRAnalyses
  PRIVATE
  KeccIR
  ${llvm_libs}
  MLIRSupport
)
