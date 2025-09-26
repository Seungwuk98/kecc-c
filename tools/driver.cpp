#include "kecc/c/ParseAST.h"
#include "kecc/driver/Compilation.h"
#include "kecc/ir/IRTransforms.h"
#include "kecc/translate/TranslatePasses.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Path.h"

namespace kecc {

namespace cl {

static llvm::cl::opt<std::string> input(llvm::cl::Positional,
                                        llvm::cl::desc("<input file>"),
                                        llvm::cl::init("-"));

static llvm::cl::opt<std::string> output("o",
                                         llvm::cl::desc("Specify output file"),
                                         llvm::cl::value_desc("filename"),
                                         llvm::cl::init("-"));

static llvm::cl::opt<bool>
    printStdOut("print-stdout",
                llvm::cl::desc("Print the output to stdout instead of a file"),
                llvm::cl::init(false));

static llvm::cl::opt<bool> emitAssembly("S", llvm::cl::desc("Emit assembly"),
                                        llvm::cl::init(false));

static llvm::cl::opt<bool> emitKecc("emit-kecc", llvm::cl::desc("Emit kecc IR"),
                                    llvm::cl::init(false));

static llvm::cl::opt<bool>
    compileOnly("c", llvm::cl::desc("Compile only, do not link"),
                llvm::cl::init(false));

static llvm::cl::opt<OptLevel>
    optLevel("O", llvm::cl::desc("Optimization level"),
             llvm::cl::values(clEnumValN(O0, "0", "No optimization"),
                              clEnumValN(O1, "1", "Optimize")),
             llvm::cl::init(O0));

struct CLOptions {

  std::function<void(ir::PassManager &)> passCallback;

  CLOptions() {
    static ir::PassPipelineParser pipelineParser("", "Passes to run\n");

    passCallback = [&](ir::PassManager &pm) {
      pipelineParser.addToPassManager(pm);
    };
  }
};

llvm::ManagedStatic<CLOptions> pmOption;

void registerPMOption() { *pmOption; }

} // namespace cl

int keccMain() {
  auto inputBufferOrErr = llvm::MemoryBuffer::getFileOrSTDIN(cl::input);
  if (inputBufferOrErr.getError()) {
    llvm::errs() << "read file error: " << inputBufferOrErr.getError().message()
                 << "\n";
    return 1;
  }

  llvm::SourceMgr sourceMgr;
  auto index =
      sourceMgr.AddNewSourceBuffer(std::move(*inputBufferOrErr), llvm::SMLoc());
  auto inputBuffer = sourceMgr.getMemoryBuffer(index);

  CompileOptTable compileOpts;

  InputFormat inputFormat;
  if (cl::input == "-") {
    inputFormat = InputFormat::C;
  } else {
    llvm::StringRef inputExt = llvm::sys::path::extension(cl::input);
    if (inputExt == ".c") {
      inputFormat = InputFormat::C;
    } else if (inputExt == ".ir") {
      inputFormat = InputFormat::KeccIR;
    } else {
      llvm::errs() << "Error: unknown input file extension: " << inputExt
                   << "\n";
      return 1;
    }
  }

  OutputFormat outputFormat;
  if (cl::output == "-") {
    outputFormat = OutputFormat::Executable;
  } else {
    llvm::StringRef outputExt = llvm::sys::path::extension(cl::output);
    if (outputExt == ".ir") {
      outputFormat = OutputFormat::KeccIR;
    } else if (outputExt == ".s") {
      outputFormat = OutputFormat::Assembly;
    } else if (outputExt == ".o") {
      outputFormat = OutputFormat::Object;
    } else {
      outputFormat = OutputFormat::Executable;
    }
  }

  if (cl::emitAssembly) {
    if (cl::emitKecc) {
      outputFormat = OutputFormat::KeccIR;
    } else {
      outputFormat = OutputFormat::Assembly;
    }
  } else {
    if (cl::emitKecc) {
      llvm::errs() << "Warning: --emit-kecc is ignored when neither -S nor -c "
                      "is given\n";
    }
    if (cl::compileOnly) {
      outputFormat = OutputFormat::Object;
    }
  }

  llvm::SmallVector<char> outputFileNameBuffer;
  if (cl::printStdOut) {
    if (cl::output != "-") {
      llvm::errs() << "Warning: --print-stdout is ignored when -o is given "
                      "with a file\n";
    }
    if (outputFormat >= OutputFormat::Object) {
      llvm::errs() << "Error: cannot print object or executable to stdout\n";
      return 1;
    }

    compileOpts.setPrintStdOut(true);
    outputFileNameBuffer.emplace_back('-');
  } else if (cl::output == "-") {
    auto ec = llvm::sys::fs::current_path(outputFileNameBuffer);
    if (ec) {
      llvm::errs() << "Error: cannot get current path: " << ec.message()
                   << "\n";
      return 1;
    }
    if (cl::input == "-") {
      llvm::sys::path::append(outputFileNameBuffer, "a");
    } else {
      outputFileNameBuffer.append(cl::input.begin(), cl::input.end());
    }
    switch (outputFormat) {
    case OutputFormat::KeccIR:
      llvm::sys::path::replace_extension(outputFileNameBuffer, ".ir");
      break;
    case OutputFormat::Assembly:
      llvm::sys::path::replace_extension(outputFileNameBuffer, ".s");
      break;
    case OutputFormat::Object:
      llvm::sys::path::replace_extension(outputFileNameBuffer, ".o");
      break;
    case OutputFormat::Executable:
      llvm::sys::path::replace_extension(outputFileNameBuffer, ".out");
      break;
    }
  } else {
    outputFileNameBuffer.append(cl::output.begin(), cl::output.end());
  }

  compileOpts.setInputFormat(inputFormat);
  compileOpts.setOutputFormat(outputFormat);
  compileOpts.setOptLevel(cl::optLevel);

  Compilation compilation(
      compileOpts, cl::input,
      llvm::StringRef{outputFileNameBuffer.data(), outputFileNameBuffer.size()},
      inputBuffer->getBuffer(), sourceMgr);

  compilation.setOptPipeline(cl::pmOption->passCallback);

  return compilation.compile();
}

} // namespace kecc

int main(int argc, char **argv) {
  llvm::InitLLVM X(argc, argv);

  kecc::ir::registerPass<kecc::ir::CanonicalizeConstant>();
  kecc::ir::registerPass<kecc::ir::Mem2Reg>();
  kecc::ir::registerPass<kecc::ir::GVN>();
  kecc::ir::registerPass<kecc::ir::DeadCode>();
  kecc::ir::registerPass<kecc::ir::OutlineConstantPass>();
  kecc::ir::registerPass<kecc::ir::InstructionFold>();
  kecc::ir::registerPass<kecc::ir::OutlineMultipleResults>();
  kecc::ir::registerPass<kecc::ir::InlineCallPass>();
  kecc::ir::registerPass<kecc::ir::CreateFunctionArgument>();
  kecc::ir::registerPass<kecc::translate::ConversionToCopyPass>();
  kecc::ir::registerPass<kecc::translate::InlineMemoryInstPass>();
  kecc::ir::registerSimplifyCFGPass();
  kecc::ir::registerCanonicalizeStructPasses();
  kecc::cl::registerPMOption();

  llvm::cl::ParseCommandLineOptions(argc, argv, "Kecc C Compiler\n");
  return kecc::keccMain();
}
