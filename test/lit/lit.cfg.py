import os
import lit.formats
from lit.llvm import llvm_config

config.name = "Kecc Integrated Test"
config.test_format = lit.formats.ShTest(True)
config.suffixes = [".ir"]

config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.kecc_bin_root, "tools")

llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)
llvm_config.with_environment("PATH", os.path.join(config.kecc_bin_root, "tools"), append_path=True)
