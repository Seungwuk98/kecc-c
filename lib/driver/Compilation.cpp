#include "kecc/driver/Compilation.h"
#include "kecc/driver/Action.h"
#include "kecc/translate/TranslatePasses.h"
namespace kecc {

TempDirectory::TempDirectory() {
  llvm::SmallVector<char> tempDir;
  llvm::sys::path::system_temp_directory(true, tempDir);
  llvm::sys::path::append(tempDir, "kecc");
  auto ec = llvm::sys::fs::createUniqueDirectory(tempDir, dir);
  assert(!ec && "failed to create temporary directory");
  (void)ec;
}

TempDirectory::~TempDirectory() {
  auto ec = llvm::sys::fs::remove_directories(dir);
  assert(!ec && "failed to remove temporary directory");
}

Compilation::Compilation(const CompileOptTable &opt,
                         llvm::StringRef inputFileName,
                         llvm::StringRef outputFileName,
                         llvm::StringRef inputSource)
    : opt(opt), inputFileName(inputFileName), outputFileName(outputFileName),
      inputSource(inputSource), irContext(new ir::IRContext) {

  auto memBuffer = llvm::MemoryBuffer::getMemBuffer(inputSource, inputFileName);
  getSourceMgr().AddNewSourceBuffer(std::move(memBuffer), llvm::SMLoc());
}

int Compilation::compile() {
  Invocation mainInvocation;

  if (opt.inputFormat == InputFormat::C) {
    mainInvocation.addAction<ParseCAction>(this); // C to IR
  } else if (opt.inputFormat == InputFormat::KeccIR) {
    mainInvocation.addAction<ParseIRAction>(this); // KeccIR source to IR
  }

  if (opt.optLevel == OptLevel::O1) {
    mainInvocation.addAction<RegisterPassesAction>(translate::addO1Passes);
  }

  if (optPipeline)
    mainInvocation.addAction<RegisterPassesAction>(optPipeline);

  mainInvocation.addAction<RunPassesAction>();

  if (opt.outputFormat == OutputFormat::Assembly) {
    mainInvocation.addAction<TranslateIRAction>(this);
  }

  llvm::SmallVector<char> outputFileBuffer;
  if (opt.printStdOut) {
    mainInvocation.addAction<PrintAction>(llvm::outs());
  } else {
    if (opt.outputFormat > OutputFormat::Assembly) {
      if (!tempDirectory)
        createTempDirectory();
      llvm::StringRef tempDir = tempDirectory->getDirectory();
      auto intputFileStem = llvm::sys::path::stem(inputFileName);
      llvm::sys::path::append(outputFileBuffer, tempDir, intputFileStem);
      llvm::sys::path::replace_extension(outputFileBuffer, ".s");
    } else {
      outputFileBuffer.append(outputFileName.begin(), outputFileName.end());
    }

    mainInvocation.addAction<OutputAction>(
        llvm::StringRef(outputFileBuffer.data(), outputFileBuffer.size()));
  }

  if (opt.outputFormat == OutputFormat::Object) {
    mainInvocation.addAction<CompileToObjAction>(
        llvm::StringRef(outputFileBuffer.data(), outputFileBuffer.size()),
        outputFileName);
  } else if (opt.outputFormat == OutputFormat::Executable) {
    mainInvocation.addAction<CompileToExeAction>(
        llvm::ArrayRef<llvm::StringRef>{
            llvm::StringRef(outputFileBuffer.data(), outputFileBuffer.size())},
        outputFileName);
  }

  auto result = mainInvocation.executeAll();
  assert(result && "Result is null");

  if (ReturnCodeResult *returnCodeResult =
          result->dyn_cast<ReturnCodeResult>()) {
    return returnCodeResult->getReturnCode();
  } else {
    return result->getLogicalResult().succeeded() ? 0 : 1;
  }
}

int Compilation::interpret(llvm::ArrayRef<llvm::StringRef> args) {
  Invocation mainInvocation;

  if (opt.inputFormat == InputFormat::C) {
    mainInvocation.addAction<ParseCAction>(this); // C to IR
  } else if (opt.inputFormat == InputFormat::KeccIR) {
    mainInvocation.addAction<ParseIRAction>(this); // KeccIR source to IR
  }

  mainInvocation.addAction<IRInterpretAction>(args);

  auto result = mainInvocation.executeAll();
  assert(result && "Result is null");
  InterpretResult *interpretResult = result->cast<InterpretResult>();

  assert(interpretResult->getReturnValues().size() <= 1);
  if (interpretResult->getReturnValues().empty())
    return 0;
  ir::VRegister *reg = interpretResult->getReturnValues()[0].get();
  assert(reg->isa<ir::VRegisterInt>());
  // cast to uint8_t to match the return of main
  return static_cast<std::uint8_t>(reg->cast<ir::VRegisterInt>()->getValue());
}

} // namespace kecc
