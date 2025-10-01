add_subdirectory(lit)
add_subdirectory(unittest)

if (KECC_ENABLED_FUZZ_TEST)
  add_subdirectory(fuzz)
endif()

