
add_library(KeccIRAnalyses
  STATIC 
  DominanceAnalysis.cpp
  VisitOrderAnalysis.cpp
  LoopAnalysis.cpp
  StructSizeAnalysis.cpp
)

target_link_libraries(KeccIRAnalyses
  PRIVATE
  KeccIR
  ${llvm_libs}
  MLIRSupport
)
