#include "kecc/driver/Action.h"
#include "kecc/driver/Compilation.h"
#include "kecc/driver/DriverConfig.h"
#include "kecc/ir/IRTransforms.h"
#include "kecc/ir/Interpreter.h"
#include "kecc/translate/TranslatePasses.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Path.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"

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

struct FuzzOptions {
  llvm::cl::OptionCategory category{"Fuzz Options"};

  llvm::cl::opt<bool> testInterpreterOnly{
      "test-interp-only",
      llvm::cl::desc("Only test the interpreter, skip native compilation"),
      llvm::cl::init(false), llvm::cl::cat(category)};
};

static llvm::ManagedStatic<FuzzOptions> fuzzOption;
void registerFuzzOption() { *fuzzOption; }

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
      auto inputStem = llvm::sys::path::stem(cl::inputOption->input);
      llvm::sys::path::append(outputFileNameBuffer, inputStem);
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

  if (cl::interpreterOption->testReturnValue.getNumOccurrences()) {
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

llvm::Expected<std::pair<int, int>>
compileNativeAndExecuteWithClangGcc(llvm::StringRef testDir) {
  // First, run clang to get the native return code
  Invocation clangInvocation;

  llvm::SmallVector<char> clangOutputFileBuffer;
  auto inputStem = llvm::sys::path::stem(cl::inputOption->input);
  llvm::sys::path::append(clangOutputFileBuffer, testDir,
                          std::format("{}_clang_native.out", inputStem.str()));
  clangInvocation.addAction<CompileToExeAction>(
      llvm::StringRef(cl::inputOption->input),
      llvm::StringRef(clangOutputFileBuffer.data(),
                      clangOutputFileBuffer.size()),
      llvm::ArrayRef<llvm::StringRef>{"-w"},
      llvm::sys::getDefaultTargetTriple());

  clangInvocation.addAction<ExecuteBinAction>(
      llvm::StringRef(clangOutputFileBuffer.data(),
                      clangOutputFileBuffer.size()),
      llvm::ArrayRef<llvm::StringRef>{}, 20);

  auto result = clangInvocation.executeAll();
  if (!result->getLogicalResult().succeeded()) {
    if (auto returnCodeResult = result->cast<ReturnCodeResult>();
        returnCodeResult->getReturnCode() < 0) {
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "Clang or binary execution timed out");
    }
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Clang invocation failed");
  }

  ReturnCodeResult *returnCodeResult = result->cast<ReturnCodeResult>();
  int nativeClangReturnCode = returnCodeResult->getReturnCode();

  // Second, run gcc to get the native return code
  Invocation gccInvocation;
  llvm::SmallVector<char> gccOutputFileBuffer;
  llvm::sys::path::append(gccOutputFileBuffer, testDir,
                          std::format("{}_gcc_native.out", inputStem.str()));
  gccInvocation.addAction<CompileToExeGccAction>(
      llvm::StringRef(cl::inputOption->input),
      llvm::StringRef(gccOutputFileBuffer.data(), gccOutputFileBuffer.size()),
      llvm::ArrayRef<llvm::StringRef>{"-w"});

  gccInvocation.addAction<ExecuteBinAction>(
      llvm::StringRef(gccOutputFileBuffer.data(), gccOutputFileBuffer.size()),
      llvm::ArrayRef<llvm::StringRef>{}, 20);
  auto gccResult = gccInvocation.executeAll();
  if (!gccResult->getLogicalResult().succeeded()) {
    if (auto returnCodeResult = gccResult->cast<ReturnCodeResult>();
        returnCodeResult->getReturnCode() < 0) {
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "GCC or binary execution timed out");
    }
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "GCC invocation failed");
  }
  ReturnCodeResult *gccReturnCodeResult = gccResult->cast<ReturnCodeResult>();
  auto nativeGCCReturnCode = gccReturnCodeResult->getReturnCode();
  if (nativeGCCReturnCode != nativeClangReturnCode) {
    llvm::errs() << "Warning: clang return code " << nativeClangReturnCode
                 << " does not match gcc's " << nativeGCCReturnCode << "\n";
  }

  return std::make_pair(nativeClangReturnCode, nativeGCCReturnCode);
}

llvm::Expected<std::pair<int, int>>
compileRiscv64AndExecuteWithClangGcc(llvm::StringRef testDir) {
  // First, run riscv64 clang to get the native return code
  Invocation riscv64ClangInvocation;

  llvm::SmallVector<char> riscv64ClangOutputFileBuffer;
  auto inputStem = llvm::sys::path::stem(cl::inputOption->input);
  llvm::sys::path::append(riscv64ClangOutputFileBuffer, testDir,
                          std::format("{}_clang_riscv64.out", inputStem.str()));
  riscv64ClangInvocation.addAction<CompileToExeAction>(
      llvm::StringRef(cl::inputOption->input),
      llvm::StringRef(riscv64ClangOutputFileBuffer.data(),
                      riscv64ClangOutputFileBuffer.size()),
      llvm::ArrayRef<llvm::StringRef>{"-w"});
  riscv64ClangInvocation.addAction<ExecuteExeByQemuAction>(
      QEMU_RISCV64_STATIC,
      llvm::StringRef(riscv64ClangOutputFileBuffer.data(),
                      riscv64ClangOutputFileBuffer.size()));
  auto riscv64ClangResult = riscv64ClangInvocation.executeAll();
  if (!riscv64ClangResult->getLogicalResult().succeeded()) {
    if (auto returnCodeResult = riscv64ClangResult->cast<ReturnCodeResult>();
        returnCodeResult->getReturnCode() < 0) {
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "riscv64 clang or binary execution timed out");
    }
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "riscv64 clang invocation failed");
  }
  auto riscv64ReturnCodeResult = riscv64ClangResult->cast<ReturnCodeResult>();
  auto riscv64ReturnCode = riscv64ReturnCodeResult->getReturnCode();

  // Second, run riscv64 gcc to get the native return code
  Invocation riscv64GccInvocation;
  llvm::SmallVector<char> riscv64GccOutputFileBuffer;
  llvm::sys::path::append(riscv64GccOutputFileBuffer, testDir,
                          std::format("{}_gcc_riscv64.out", inputStem.str()));
  riscv64GccInvocation.addAction<CompileToExeGccAction>(
      llvm::StringRef(cl::inputOption->input),
      llvm::StringRef(riscv64GccOutputFileBuffer.data(),
                      riscv64GccOutputFileBuffer.size()),
      llvm::ArrayRef<llvm::StringRef>{"-w"}, RISCV64_GCC_DIR);
  riscv64GccInvocation.addAction<ExecuteExeByQemuAction>(
      QEMU_RISCV64_STATIC, llvm::StringRef(riscv64GccOutputFileBuffer.data(),
                                           riscv64GccOutputFileBuffer.size()));
  auto riscv64GccResult = riscv64GccInvocation.executeAll();
  if (!riscv64GccResult->getLogicalResult().succeeded()) {
    if (auto returnCodeResult = riscv64GccResult->cast<ReturnCodeResult>();
        returnCodeResult->getReturnCode() < 0) {
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "riscv64 gcc or binary execution timed out");
    }
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "riscv64 gcc invocation failed");
  }
  auto riscv64GccReturnCodeResult = riscv64GccResult->cast<ReturnCodeResult>();
  auto riscv64GccReturnCode = riscv64GccReturnCodeResult->getReturnCode();
  if (riscv64GccReturnCode != riscv64ReturnCode) {
    llvm::errs() << "Warning: riscv64 clang return code " << riscv64ReturnCode
                 << " does not match riscv64 gcc's " << riscv64GccReturnCode
                 << "\n";
  }
  return std::make_pair(riscv64ReturnCode, riscv64GccReturnCode);
}

int fuzzMain() {
  if (cl::inputOption->input == "-") {
    llvm::errs() << "Error: input file must be specified\n";
    return 1;
  }

  auto inputBufferOrErr = llvm::MemoryBuffer::getFile(cl::inputOption->input);
  if (inputBufferOrErr.getError()) {
    llvm::errs() << "read file error: " << inputBufferOrErr.getError().message()
                 << "\n";
    return 1;
  }

  InputFormat inputFormat;
  llvm::StringRef inputExt = llvm::sys::path::extension(cl::inputOption->input);
  if (inputExt == ".c") {
    inputFormat = InputFormat::C;
  } else {
    llvm::errs() << "Error: unknown input file extension for fuzz: " << inputExt
                 << "\n";
    return 1;
  }

  llvm::SmallVector<char> testDir;
  {
    llvm::StringRef parentDir =
        llvm::sys::path::parent_path(cl::inputOption->input);
    testDir.append(parentDir.begin(), parentDir.end());
    if (!llvm::sys::path::is_absolute(parentDir)) {
      llvm::SmallVector<char> currentPath;
      if (auto ec = llvm::sys::fs::current_path(currentPath)) {
        llvm::errs() << "Error: cannot get current path: " << ec.message()
                     << "\n";
        return 1;
      }
      llvm::SmallVector<char> absTestDir;
      llvm::sys::path::append(absTestDir, currentPath);
      llvm::sys::path::append(absTestDir, testDir);
      llvm::sys::path::remove_dots(absTestDir);
      testDir = std::move(absTestDir);
    }
  }

  CompileOptTable compileOpts;
  compileOpts.setInputFormat(inputFormat);

  Compilation compilation(compileOpts, cl::inputOption->input, "-",
                          inputBufferOrErr->get()->getBuffer());
  compilation.setOptPipeline(cl::pmOption->passCallback);

  auto nativeReturnCodesOrErr = compileNativeAndExecuteWithClangGcc(
      llvm::StringRef(testDir.data(), testDir.size()));
  if (auto error = nativeReturnCodesOrErr.takeError()) {
    llvm::errs() << "Error: " << toString(std::move(error)) << "\n";
    return 0;
  }
  auto [nativeClangReturnCode, nativeGCCReturnCode] = *nativeReturnCodesOrErr;

  // Then, run interpreter to get the kecc interpreted return code
  Invocation fullInvocation;
  Invocation interpretInvocation;

  fullInvocation.addAction<ParseCAction>(&compilation, true);
  interpretInvocation.addAction<ParseCAction>(&compilation, true);

  llvm::StringRef inputStem = llvm::sys::path::stem(cl::inputOption->input);
  llvm::SmallVector<char> interpretIRFileBuffer;
  llvm::sys::path::append(interpretIRFileBuffer, testDir,
                          std::format("{}_interp_kecc.ir", inputStem.str()));
  interpretInvocation.addAction<OutputAction>(llvm::StringRef(
      interpretIRFileBuffer.data(), interpretIRFileBuffer.size()));
  interpretInvocation.addAction<IRInterpretAction>(std::nullopt);
  auto interpretInvokeResult = interpretInvocation.executeAll();
  if (!interpretInvokeResult->getLogicalResult().succeeded()) {
    llvm::errs() << "Error: interpretation failed\n";
    return 1;
  }

  InterpretResult *interpretResult =
      interpretInvokeResult->cast<InterpretResult>();
  assert(interpretResult->getReturnValues().size() == 1 &&
         "main function should return one integer value");
  int interpretReturnCode =
      static_cast<uint8_t>(interpretResult->getReturnValues()[0]
                               ->cast<ir::VRegisterInt>()
                               ->getValue());
  bool testFailed = false;
  if (interpretReturnCode == nativeClangReturnCode ||
      interpretReturnCode == nativeGCCReturnCode) {
    if (interpretReturnCode != nativeClangReturnCode)
      llvm::errs() << "Warning: interpreted return code " << interpretReturnCode
                   << " matches gcc's " << nativeGCCReturnCode
                   << ", but not clang's " << nativeClangReturnCode << "\n";
    else if (interpretReturnCode != nativeGCCReturnCode)
      llvm::errs() << "Warning: interpreted return code " << interpretReturnCode
                   << " matches clang's " << nativeClangReturnCode
                   << ", but not gcc's " << nativeGCCReturnCode << "\n";
    else
      llvm::outs() << "Info: interpreted return code " << interpretReturnCode
                   << " matches both clang and gcc\n";
  } else {
    llvm::errs() << "Error: interpreted return code " << interpretReturnCode
                 << " does not match clang's " << nativeClangReturnCode
                 << " nor gcc's " << nativeGCCReturnCode << "\n";
    testFailed = true;
  }

  if (cl::fuzzOption->testInterpreterOnly) {
    return testFailed;
  }

  auto riscv64ReturnCodesOrErr = compileRiscv64AndExecuteWithClangGcc(
      llvm::StringRef(testDir.data(), testDir.size()));
  if (auto error = riscv64ReturnCodesOrErr.takeError()) {
    llvm::errs() << "Error: " << toString(std::move(error)) << "\n";
    return 0;
  }

  auto [riscv64ClangReturnCode, riscv64GCCReturnCode] =
      *riscv64ReturnCodesOrErr;

  // Finally, do the full kecc compilation and execution
  fullInvocation.addAction<RegisterPassesAction>(compilation.getOptPipeline());
  fullInvocation.addAction<RunPassesAction>();

  llvm::SmallVector<char> irFileBuffer;
  llvm::sys::path::append(irFileBuffer, testDir,
                          std::format("{}_kecc.ir", inputStem.str()));

  fullInvocation.addAction<OutputAction>(
      llvm::StringRef(irFileBuffer.data(), irFileBuffer.size()));

  fullInvocation.addAction<RegisterPassesAction>([](ir::PassManager &pm) {
    translate::registerDefaultTranslationPasses(pm);
  });
  fullInvocation.addAction<RunPassesAction>();
  fullInvocation.addAction<TranslateIRAction>(&compilation);

  llvm::SmallVector<char> asmFileBuffer;
  llvm::sys::path::append(asmFileBuffer, testDir,
                          std::format("{}_kecc.s", inputStem.str()));
  fullInvocation.addAction<OutputAction>(
      llvm::StringRef(asmFileBuffer.data(), asmFileBuffer.size()));

  llvm::SmallVector<char> keccOutputFileBuffer;
  llvm::sys::path::append(keccOutputFileBuffer, testDir,
                          std::format("{}_kecc.out", inputStem.str()));

  fullInvocation.addAction<CompileToExeAction>(
      llvm::StringRef(asmFileBuffer.data(), asmFileBuffer.size()),
      llvm::StringRef(keccOutputFileBuffer.data(),
                      keccOutputFileBuffer.size()));

  fullInvocation.addAction<ExecuteExeByQemuAction>(
      QEMU_RISCV64_STATIC, llvm::StringRef(keccOutputFileBuffer.data(),
                                           keccOutputFileBuffer.size()));

  auto fullResult = fullInvocation.executeAll();
  if (!fullResult->getLogicalResult().succeeded()) {
    llvm::errs() << "Error: full invocation failed\n";
    return 1;
  }
  ReturnCodeResult *fullReturnCodeResult = fullResult->cast<ReturnCodeResult>();
  int fullReturnCode =
      static_cast<uint8_t>(fullReturnCodeResult->getReturnCode());

  if (fullReturnCode == riscv64ClangReturnCode ||
      fullReturnCode == riscv64GCCReturnCode) {
    if (fullReturnCode != riscv64ClangReturnCode)
      llvm::errs() << "Warning: full compiled return code " << fullReturnCode
                   << " matches riscv64 gcc's " << riscv64GCCReturnCode
                   << ", but not riscv64 clang's " << riscv64ClangReturnCode
                   << "\n";
    else if (fullReturnCode != riscv64GCCReturnCode)
      llvm::errs() << "Warning: full compiled return code " << fullReturnCode
                   << " matches riscv64 clang's " << riscv64ClangReturnCode
                   << ", but not riscv64 gcc's " << riscv64GCCReturnCode
                   << "\n";
    else
      llvm::outs() << "Info: full compiled return code " << fullReturnCode
                   << " matches both riscv64 clang and gcc\n";
  } else {
    llvm::errs() << "Error: full compiled return code " << fullReturnCode
                 << " does not match riscv64 clang's " << riscv64ClangReturnCode
                 << " nor riscv64 gcc's " << riscv64GCCReturnCode << "\n";
    testFailed = true;
  }

  return testFailed;
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

int fuzz_main(int argc, char **argv) {
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
  kecc::cl::registerPMOption();
  kecc::cl::registerFuzzOption();

  llvm::cl::ParseCommandLineOptions(argc, argv, "Kecc C Compiler\n");
  return kecc::fuzzMain();
}
