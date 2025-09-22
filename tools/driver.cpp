#include "kecc/c/ParseAST.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"

namespace kecc {

namespace cl {

static llvm::cl::opt<std::string> input(llvm::cl::Positional,
                                        llvm::cl::desc("<input file>"),
                                        llvm::cl::init("-"));

static llvm::cl::opt<std::string> output("o",
                                         llvm::cl::desc("Specify output file"),
                                         llvm::cl::value_desc("filename"),
                                         llvm::cl::init("-"));

static llvm::cl::opt<bool> emitAssembly("S", llvm::cl::desc("Emit assembly"),
                                        llvm::cl::init(false));

static llvm::cl::opt<bool> emitKecc("emit-kecc", llvm::cl::desc("Emit kecc IR"),
                                    llvm::cl::init(false));

static llvm::cl::opt<bool>
    compileOnly("c", llvm::cl::desc("Compile only, do not link"),
                llvm::cl::init(false));

enum Opt {
  O0,
  O1,
};

static llvm::cl::opt<Opt>
    optLevel("O", llvm::cl::desc("Optimization level"),
             llvm::cl::values(clEnumValN(O0, "0", "No optimization"),
                              clEnumValN(O1, "1", "Optimize")),
             llvm::cl::init(O0));

} // namespace cl

int keccMain() { return 0; }

} // namespace kecc

int main(int argc, char **argv) {
  llvm::InitLLVM X(argc, argv);

  return kecc::keccMain();
}
