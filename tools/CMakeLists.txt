add_executable(kecc-opt driver.cpp)

target_link_libraries(
  kecc-opt
  PRIVATE
  ${llvm_libs}  
  KeccParser
  KeccIRAnalyses
  KeccIRPasses
)
