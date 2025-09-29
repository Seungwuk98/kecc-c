get_property(LLVM_LIT_CONFIG_FILES GLOBAL PROPERTY LLVM_LIT_CONFIG_FILES)
list(LENGTH LLVM_LIT_CONFIG_FILES file_len)
math(EXPR file_last "${file_len} - 1")

get_llvm_lit_path(LIT_BASE_DIR LIT_FILE_NAME ALLOW_EXTERNAL)

set(LLVM_LIT_CONFIG_MAP "${LLVM_LIT_PATH_FUNCTION}\n")
if (${file_last} GREATER -1)
  foreach(i RANGE 0 ${file_last} 2)
    list(GET LLVM_LIT_CONFIG_FILES ${i} main_config)
    math(EXPR i1 "${i} + 1")
    list(GET LLVM_LIT_CONFIG_FILES ${i1} site_out)
    set(map "map_config(path(r'${main_config}'), path(r'${site_out}'))")
    set(LLVM_LIT_CONFIG_MAP "${LLVM_LIT_CONFIG_MAP}\n${map}")
  endforeach()
endif()


configure_file(
  llvm-lit.py.in
  ${LIT_BASE_DIR}/${LIT_FILE_NAME}
)
