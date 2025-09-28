
macro(add_kecc_driver name) 
  cmake_parse_arguments(ARG "GEN_DRIVER_MAIN" "" "SOURCES;DEPENDS" ${ARGN})

  if (${ARG_GEN_DRIVER_MAIN})
    set(KECC_DRIVER_NAME ${name}) 
    set(KECC_GENERATED_DRIVER ${CMAKE_CURRENT_BINARY_DIR}/${name}-driver.cpp) 
    configure_file(
      ${PROJECT_SOURCE_DIR}/cmake/kecc-driver-template.cpp.in 
      ${KECC_GENERATED_DRIVER} @ONLY)
    add_executable(${name} ${KECC_GENERATED_DRIVER} ${ARG_SOURCES})
    if (${ARG_DEPENDS})
      add_dependencies(${name} ${ARG_DEPENDS})  
    endif()
  endif()
endmacro()
