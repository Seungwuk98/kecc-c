add_executable(kecc-opt kecc-opt.cpp)

target_link_libraries(
  kecc-opt
  PRIVATE
  ${llvm_libs}  
  KeccParser
  KeccIRAnalyses
  KeccIRPasses
  KeccTranslate
)

add_executable(kecc-translate kecc-translate.cpp)

target_link_libraries(
  kecc-translate
  PRIVATE
  ${llvm_libs}
  KeccIR
  KeccParser
  KeccIRAnalyses
  KeccIRPasses
  KeccASM
  KeccTranslate
)
