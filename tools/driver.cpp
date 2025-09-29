#include "kecc/driver/Compilation.h"
#include "kecc/ir/IRTransforms.h"
#include "kecc/translate/TranslatePasses.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Path.h"

namespace kecc {

namespace cl {

static llvm::cl::OptionCategory category{"Input/Output Options"};

struct InputCLOptions {
  llvm::cl::opt<std::string> input{
      llvm::cl::Positional, llvm::cl::desc("<input file>"), llvm::cl::init("-"),
      llvm::cl::value_desc("filename"), llvm::cl::cat(category)};

  llvm::cl::opt<InputFormat> inputFormat{
      "input-format", llvm::cl::desc("Specify input format"),
      llvm::cl::values(clEnumValN(InputFormat::C, "c", "C source code"),
                       clEnumValN(InputFormat::KeccIR, "ir",
                                  "Kecc Intermediate Representation")),
      llvm::cl::init(InputFormat::C), llvm::cl::cat(category)};
};

static llvm::ManagedStatic<InputCLOptions> inputOption;

struct OutputCLOptions {
  llvm::cl::opt<std::string> output{"o", llvm::cl::desc("Specify output file"),
                                    llvm::cl::value_desc("filename"),
                                    llvm::cl::init("-"),
                                    llvm::cl::cat(category)};
  llvm::cl::opt<bool> printStdOut{
      "print-stdout",
      llvm::cl::desc("Print the output to stdout instead of a file"),
      llvm::cl::init(false)};
};

static llvm::ManagedStatic<OutputCLOptions> outputOption;

void registerInputOption() { *inputOption; }
void registerOutputOption() { *outputOption; }

struct OutputFormatOptions {
  llvm::cl::OptionCategory category{"Output Format Options"};

  llvm::cl::opt<bool> emitAssembly{"S", llvm::cl::desc("Emit assembly"),
                                   llvm::cl::init(false),
                                   llvm::cl::cat(category)};

  llvm::cl::opt<bool> emitKecc{"emit-kecc", llvm::cl::desc("Emit kecc IR"),
                               llvm::cl::init(false), llvm::cl::cat(category)};

  llvm::cl::opt<bool> compileOnly{
      "c", llvm::cl::desc("Compile only, do not link"), llvm::cl::init(false),
      llvm::cl::cat(category)};

  llvm::cl::opt<OptLevel> optLevel{
      "O", llvm::cl::desc("Optimization level"),
      llvm::cl::values(clEnumValN(O0, "0", "No optimization"),
                       clEnumValN(O1, "1", "Optimize")),
      llvm::cl::init(O0), llvm::cl::cat(category)};
};

llvm::ManagedStatic<OutputFormatOptions> outputFormatOption;
void registerOutputFormatOption() { *outputFormatOption; }

struct InterpreterOptions {
  llvm::cl::OptionCategory category{"Interpreter Options"};

  llvm::cl::opt<int> testReturnValue{
      "test-return-value",
      llvm::cl::desc("Test the return value of main function"),
      llvm::cl::init(0), llvm::cl::cat(category)};

  enum PrintReturnValueOption { PRINT_PREFIX, ONLY_VALUE };

  llvm::cl::opt<PrintReturnValueOption> printReturnValue{
      "print-return-value",
      llvm::cl::desc("Print the return value of main function"),
      llvm::cl::values(
          clEnumValN(PRINT_PREFIX, "with-prefix",
                     "Print with 'Return value: ' prefix"),
          clEnumValN(ONLY_VALUE, "only-value", "Print only value")),
      llvm::cl::init(PRINT_PREFIX), llvm::cl::cat(category)};

  llvm::cl::opt<std::string> mainArguments{
      "main-args",
      llvm::cl::desc("Arguments passed to main function, separated by space"),
      llvm::cl::init(""), llvm::cl::cat(category)};
};

llvm::ManagedStatic<InterpreterOptions> interpreterOption;
void registerInterpreterOption() { *interpreterOption; }

struct PassOptions {

  std::function<void(ir::PassManager &)> passCallback;

  PassOptions() {
    static ir::PassPipelineParser pipelineParser("", "Passes to run\n");

    passCallback = [&](ir::PassManager &pm) {
      pipelineParser.addToPassManager(pm);
    };
  }
};

static llvm::ManagedStatic<PassOptions> pmOption;

void registerPMOption() { *pmOption; }

} // namespace cl

int keccMain() {
  auto inputBufferOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(cl::inputOption->input);
  if (inputBufferOrErr.getError()) {
    llvm::errs() << "read file error: " << inputBufferOrErr.getError().message()
                 << "\n";
    return 1;
  }

  CompileOptTable compileOpts;

  InputFormat inputFormat;
  if (cl::inputOption->inputFormat.getNumOccurrences()) {
    inputFormat = cl::inputOption->inputFormat;
  } else if (cl::inputOption->input == "-") {
    inputFormat = InputFormat::KeccIR;
  } else {
    llvm::StringRef inputExt =
        llvm::sys::path::extension(cl::inputOption->input);
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
  if (cl::outputOption->output == "-" &&
      !cl::outputOption->output.getNumOccurrences()) {
    outputFormat = OutputFormat::Executable;
  } else {
    llvm::StringRef outputExt =
        llvm::sys::path::extension(cl::outputOption->output);
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

  if (cl::outputFormatOption->emitAssembly) {
    if (cl::outputFormatOption->emitKecc) {
      outputFormat = OutputFormat::KeccIR;
    } else {
      outputFormat = OutputFormat::Assembly;
    }

    if (cl::outputOption->output == "-" &&
        cl::outputOption->output.getNumOccurrences()) {
      cl::outputOption->printStdOut.setValue(true);
    }
  } else {
    if (cl::outputFormatOption->emitKecc) {
      llvm::errs() << "Warning: --emit-kecc is ignored when neither -S nor -c "
                      "is given\n";
    }
    if (cl::outputFormatOption->compileOnly) {
      outputFormat = OutputFormat::Object;
    }
  }

  llvm::SmallVector<char> outputFileNameBuffer;
  if (cl::outputOption->printStdOut) {
    if (cl::outputOption->output != "-") {
      llvm::errs() << "Warning: --print-stdout is ignored when -o is given "
                      "with a file\n";
    }
    if (outputFormat >= OutputFormat::Object) {
      llvm::errs() << "Error: cannot print object or executable to stdout\n";
      return 1;
    }

    compileOpts.setPrintStdOut(true);
    outputFileNameBuffer.emplace_back('-');
  } else if (cl::outputOption->output == "-") {
    auto ec = llvm::sys::fs::current_path(outputFileNameBuffer);
    if (ec) {
      llvm::errs() << "Error: cannot get current path: " << ec.message()
                   << "\n";
      return 1;
    }
    if (cl::inputOption->input == "-") {
      llvm::sys::path::append(outputFileNameBuffer, "a");
    } else {
      outputFileNameBuffer.append(cl::inputOption->input.begin(),
                                  cl::inputOption->input.end());
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
    outputFileNameBuffer.append(cl::outputOption->output.begin(),
                                cl::outputOption->output.end());
  }

  compileOpts.setInputFormat(inputFormat);
  compileOpts.setOutputFormat(outputFormat);
  compileOpts.setOptLevel(cl::outputFormatOption->optLevel);

  Compilation compilation(
      compileOpts, cl::inputOption->input,
      llvm::StringRef{outputFileNameBuffer.data(), outputFileNameBuffer.size()},
      inputBufferOrErr->get()->getBuffer());

  compilation.setOptPipeline(cl::pmOption->passCallback);

  return compilation.compile();
}

int keciMain() {
  auto inputBufferOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(cl::inputOption->input);
  if (inputBufferOrErr.getError()) {
    llvm::errs() << "read file error: " << inputBufferOrErr.getError().message()
                 << "\n";
    return 1;
  }

  CompileOptTable compileOpts;

  InputFormat inputFormat;
  if (cl::inputOption->inputFormat.getNumOccurrences())
    inputFormat = cl::inputOption->inputFormat;
  else if (cl::inputOption->input == "-") {
    inputFormat = InputFormat::KeccIR;
  } else {
    llvm::StringRef inputExt =
        llvm::sys::path::extension(cl::inputOption->input);
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

  compileOpts.setInputFormat(inputFormat);

  Compilation compilation(compileOpts, cl::inputOption->input, "-",
                          inputBufferOrErr->get()->getBuffer());

  llvm::SmallVector<llvm::StringRef> mainArgs;
  mainArgs.emplace_back("keci");
  if (cl::interpreterOption->mainArguments.getNumOccurrences()) {
    llvm::StringRef argsStr = cl::interpreterOption->mainArguments;
    argsStr.split(mainArgs, ' ', -1, false);
  }

  int returnValue = compilation.interpret(mainArgs);

  if (cl::interpreterOption->printReturnValue.getNumOccurrences()) {
    if (cl::interpreterOption->printReturnValue ==
        cl::InterpreterOptions::PRINT_PREFIX) {
      llvm::outs() << "Return value: " << returnValue << "\n";
    } else {
      llvm::outs() << returnValue << '\n';
    }
  }

  if (cl::interpreterOption->testReturnValue) {
    if (returnValue != cl::interpreterOption->testReturnValue) {
      llvm::errs() << "Error: return value " << returnValue
                   << " does not match expected "
                   << cl::interpreterOption->testReturnValue << "\n";
      return 1;
    }
    return 0;
  }

  return returnValue;
}

} // namespace kecc

int kecc_main(int argc, char **argv) {
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
  kecc::cl::registerInputOption();
  kecc::cl::registerOutputOption();
  kecc::cl::registerOutputFormatOption();
  kecc::cl::registerPMOption();

  llvm::cl::ParseCommandLineOptions(argc, argv, "Kecc C Compiler\n");
  return kecc::keccMain();
}

int keci_main(int argc, char **argv) {
  kecc::cl::registerInputOption();
  kecc::cl::registerInterpreterOption();
  llvm::cl::ParseCommandLineOptions(argc, argv, "Kecc IR Interpreter\n");
  return kecc::keciMain();
}
