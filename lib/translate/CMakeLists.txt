add_subdirectory(TranslationRules)

add_library(
  KeccTranslate
  STATIC 
  IRTranslater.cpp
  InterferenceGraph.cpp
  SpillAnalysis.cpp
  TranslateContext.cpp
  LiveRangeAnalysis.cpp
  LivenessAnalysis.cpp
)

target_link_libraries(KeccTranslate
  PRIVATE
  ${llvm_libs}
  MLIRSupport
  KeccIR
  KeccASM
  KeccTranslationRules
)
