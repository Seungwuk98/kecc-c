

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in 
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG 
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(KECC_TEST_DEPENDS
  kecc-opt
  kecc-translate
  kecc-test-executor
  kecc
  keci
)

find_program(Python3_EXECUTABLE python3 REQUIRED)
message(STATUS "Found Python3: ${Python3_EXECUTABLE}")

add_lit_testsuite(check-kecc "Running the kecc testsuite"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${KECC_TEST_DEPENDS}
  ARGS --timeout=30
)

set_target_properties(check-kecc PROPERTIES FOLDER "Kecc Tests")

add_lit_testsuites(
  KECC ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${KECC_TEST_DEPENDS}
  ARGS --timeout=30
)
