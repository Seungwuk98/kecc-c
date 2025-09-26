#include "kecc/driver/Compilation.h"
#include "kecc/asm/Asm.h"
#include "kecc/c/Diag.h"
#include "kecc/c/IRGenerator.h"
#include "kecc/c/ParseAST.h"
#include "kecc/driver/Action.h"
#include "kecc/driver/DriverConfig.h"
#include "kecc/driver/DriverConfig.h.in"
#include "kecc/ir/IR.h"
#include "kecc/ir/IRAnalyses.h"
#include "kecc/ir/Instruction.h"
#include "kecc/ir/Pass.h"
#include "kecc/parser/Lexer.h"
#include "kecc/parser/Parser.h"
#include "kecc/translate/IRTranslater.h"
#include "kecc/translate/TranslatePasses.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/Program.h"
#include <format>
#include <memory>

namespace kecc {

class OptArg : public ActionDataTemplate<OptArg> {
public:
  OptArg(utils::LogicalResult result, std::unique_ptr<ir::Module> module)
      : Base(result), module(std::move(module)),
        passManager(new ir::PassManager), recordDeclManager(nullptr) {}

  static llvm::StringRef getNameStr() { return "Optimization Argument"; }
  llvm::StringRef getName() const override { return getNameStr(); }

  void setModule(std::unique_ptr<ir::Module> mod) { module = std::move(mod); }
  ir::Module *getModule() const { return module.get(); }
  ir::PassManager &getPassManager() { return *passManager; }

  void createRecordDeclManager() {
    recordDeclManager = std::make_unique<c::RecordDeclManager>();
  }
  c::RecordDeclManager *getRecordDeclManager() const {
    return recordDeclManager.get();
  }

private:
  std::unique_ptr<ir::Module> module;
  std::unique_ptr<ir::PassManager> passManager;
  std::unique_ptr<c::RecordDeclManager> recordDeclManager;
};

class AsmArg : public ActionDataTemplate<AsmArg> {
public:
  AsmArg(utils::LogicalResult result, std::unique_ptr<as::Asm> asmModule)
      : Base(result), asmModule(std::move(asmModule)) {}

  static llvm::StringRef getNameStr() { return "Assembly Argument"; }
  llvm::StringRef getName() const override { return getNameStr(); }

  as::Asm *getAsm() const { return asmModule.get(); }
  void setAsm(std::unique_ptr<as::Asm> asmMod) {
    asmModule = std::move(asmMod);
  }

private:
  std::unique_ptr<as::Asm> asmModule;
};

class ReturnCodeResult : public ActionDataTemplate<ReturnCodeResult> {
public:
  ReturnCodeResult(utils::LogicalResult result, int returnCode)
      : Base(result), returnCode(returnCode) {}

  static llvm::StringRef getNameStr() { return "Return Code Result"; }
  llvm::StringRef getName() const override { return getNameStr(); }

  int getReturnCode() const { return returnCode; }

private:
  int returnCode;
};

class ParseCAction : public ActionTemplate<ActionData, OptArg> {
public:
  ParseCAction(Compilation *compilation) : Base(compilation) {}

  static llvm::StringRef getNameStr() { return "Parse C source"; }
  llvm::StringRef getActionName() const override { return getNameStr(); }

  void preExecute(ActionData *) override {
    if (!getCompilation()->getIRContext())
      getCompilation()->createIRContext();
  }

  std::unique_ptr<ActionData> execute(std::unique_ptr<ActionData>) override {
    auto &compilation = *getCompilation();

    c::ParseAST parser(compilation.getInputSource(),
                       compilation.getInputFileName());
    auto result = parser.parse();
    if (!result.succeeded())
      return std::make_unique<OptArg>(utils::LogicalResult::failure(), nullptr);

    auto astUnit = parser.getASTUnit();

    c::ClangDiagManager diagManager(&astUnit->getDiagnostics(), astUnit);

    std::unique_ptr<OptArg> optArg =
        std::make_unique<OptArg>(utils::LogicalResult::success(), nullptr);

    optArg->createRecordDeclManager();
    ir::IR *ir = new ir::IR(compilation.getIRContext());
    c::IRGenerator irGen(&astUnit->getASTContext(), &diagManager,
                         *optArg->getRecordDeclManager(), ir);

    irGen.VisitTranslationUnitDecl(
        astUnit->getASTContext().getTranslationUnitDecl());
    if (diagManager.hasError())
      return std::make_unique<OptArg>(utils::LogicalResult::failure(), nullptr);

    auto module = ir::Module::create(std::unique_ptr<ir::IR>(ir));
    module->getOrCreateAnalysis<ir::StructSizeAnalysis>(
        module.get(), optArg->getRecordDeclManager()->getStructSizeMap(),
        optArg->getRecordDeclManager()->getStructFieldsMap());
    optArg->setModule(std::move(module));
    return optArg;
  }
};

class ParseIRAction : public ActionTemplate<ActionData, OptArg> {
public:
  ParseIRAction(Compilation *compilation) : Base(compilation) {}

  static llvm::StringRef getNameStr() { return "Parse IR source"; }
  llvm::StringRef getActionName() const override { return getNameStr(); }

  void preExecute(ActionData *) override {
    if (!getCompilation()->getIRContext())
      getCompilation()->createIRContext();
  }

  std::unique_ptr<ActionData> execute(std::unique_ptr<ActionData>) override {
    auto &compilation = *getCompilation();

    Lexer lexer(compilation.getInputSource(), compilation.getIRContext());
    Parser parser(lexer, compilation.getIRContext());

    std::unique_ptr<ir::Module> module = parser.parseAndBuildModule();
    if (compilation.getIRContext()->diag().hasError())
      return std::make_unique<OptArg>(utils::LogicalResult::failure(), nullptr);

    return std::make_unique<OptArg>(utils::LogicalResult::success(),
                                    std::move(module));
  }
};

class RegisterPassesAction : public ActionTemplate<OptArg, OptArg> {
public:
  RegisterPassesAction(Compilation *compilation,
                       llvm::ArrayRef<ir::Pass *> passes)
      : ActionTemplate(compilation), passes(passes) {}

  RegisterPassesAction(Compilation *compilation,
                       std::function<void(ir::PassManager &)> passRegisterFunc)
      : ActionTemplate(compilation), passes(std::move(passRegisterFunc)) {}

  static llvm::StringRef getNameStr() { return "Register Passes"; }
  llvm::StringRef getActionName() const override { return getNameStr(); }

  std::unique_ptr<OptArg> execute(std::unique_ptr<OptArg> arg) override {
    auto &pm = arg->getPassManager();
    std::visit(
        [&]<typename T>(T &&passes) {
          using PassesType = std::decay_t<T>;
          if constexpr (std::is_same_v<PassesType,
                                       llvm::ArrayRef<ir::Pass *>>) {
            for (ir::Pass *pass : passes) {
              assert(pass && "Pass is null");
              pm.addPass(pass);
            }
          } else {
            passes(pm);
          }
        },
        passes);

    return arg;
  }

private:
  std::variant<llvm::ArrayRef<ir::Pass *>,
               std::function<void(ir::PassManager &)>>
      passes;
};

class RunPassesAction : public ActionTemplate<OptArg, OptArg> {
public:
  RunPassesAction(Compilation *compilation) : Base(compilation) {}

  static llvm::StringRef getNameStr() { return "Run Passes"; }
  llvm::StringRef getActionName() const override { return getNameStr(); }

  std::unique_ptr<OptArg> execute(std::unique_ptr<OptArg> arg) override {
    auto &pm = arg->getPassManager();
    auto module = arg->getModule();
    auto result = pm.run(module);
    if (result.isFailure())
      return std::make_unique<OptArg>(utils::LogicalResult::failure(), nullptr);
    return arg;
  }
};

class TranslateIRAction : public ActionTemplate<OptArg, AsmArg> {
public:
  TranslateIRAction(Compilation *compilation) : Base(compilation) {}

  static llvm::StringRef getNameStr() { return "Translate to Assembly"; }
  llvm::StringRef getActionName() const override { return getNameStr(); }

  void preExecute(OptArg *) override {
    if (!getCompilation()->getTranslateContext())
      getCompilation()->createTranslateContext();

    defaultRegisterSetup(getCompilation()->getTranslateContext());
    registerDefaultTranslationRules(getCompilation()->getTranslateContext());
  }

  std::unique_ptr<AsmArg> execute(std::unique_ptr<OptArg> arg) override {
    TranslateContext *translateContext =
        getCompilation()->getTranslateContext();
    auto *spillAnalysis = arg->getModule()->getOrCreateAnalysis<SpillAnalysis>(
        arg->getModule(), translateContext);
    assert(spillAnalysis && "SpillAnalysis is not available");

    spillAnalysis->spillFull();
    IRTranslater irTranslater(translateContext, arg->getModule());
    auto asmModule = irTranslater.translate();
    if (!asmModule)
      return std::make_unique<AsmArg>(utils::LogicalResult::failure(), nullptr);
    return std::make_unique<AsmArg>(utils::LogicalResult::success(),
                                    std::move(asmModule));
  }
};

class OutputAction : public Action {
public:
  OutputAction(Compilation *compilation, llvm::StringRef output)
      : Action(compilation) {}

  static llvm::StringRef getNameStr() { return "Output Assembly"; }
  llvm::StringRef getActionName() const override { return getNameStr(); }

  std::unique_ptr<ActionData>
  execute(std::unique_ptr<ActionData> arg) override {
    std::error_code ec;
    llvm::raw_fd_ostream os(outputFileName, ec);
    if (ec) {
      llvm::errs() << "Error opening output file: " << ec.message() << "\n";
      return std::make_unique<ActionData>(utils::LogicalResult::failure());
    }

    if (AsmArg *asmArg = llvm::dyn_cast<AsmArg>(arg.get())) {
      asmArg->getAsm()->print(os);
    } else if (OptArg *optArg = llvm::dyn_cast<OptArg>(arg.get())) {
      ir::IRPrintContext printContext(os);
      optArg->getModule()->getIR()->print(printContext);
    } else {
      llvm_unreachable("Argument is neither OptArg nor AsmArg");
    }
    return std::make_unique<ActionData>(utils::LogicalResult::success());
  }

private:
  llvm::StringRef outputFileName;
};

class PrintAction : public Action {
public:
  PrintAction(Compilation *compilation, llvm::raw_ostream &os)
      : Action(compilation), os(&os) {}

  static llvm::StringRef getNameStr() { return "Print Action"; }
  llvm::StringRef getActionName() const override { return getNameStr(); }

  std::unique_ptr<ActionData>
  execute(std::unique_ptr<ActionData> arg) override {
    if (AsmArg *asmArg = llvm::dyn_cast<AsmArg>(arg.get())) {
      asmArg->getAsm()->print(*os);
    } else if (OptArg *optArg = llvm::dyn_cast<OptArg>(arg.get())) {
      ir::IRPrintContext printContext(*os);
      optArg->getModule()->getIR()->print(printContext);
    } else {
      llvm_unreachable("Argument is neither OptArg nor AsmArg");
    }
    return std::make_unique<ActionData>(utils::LogicalResult::success());
  }

private:
  llvm::raw_ostream *os;
};

class ClangExecutor {
public:
  ClangExecutor() { arguments.emplace_back(CLANG_DIR); }

  void addArgument(llvm::StringRef arg) { arguments.emplace_back(arg); }

  void addArgument(llvm::StringRef argName, llvm::StringRef value) {
    if (value.empty())
      addArgument(argName);
    arguments.emplace_back(std::format("{}={}", argName.str(), value.str()));
  }

  void addArgument(llvm::DenseMap<llvm::StringRef, llvm::StringRef> args) {
    for (auto &[argName, value] : args) {
      addArgument(argName, value);
    }
  }

  int execute() {
    llvm::SmallVector<llvm::StringRef> argRefs = llvm::map_to_vector(
        arguments, [](const std::string &s) { return llvm::StringRef(s); });

    return llvm::sys::ExecuteAndWait(CLANG_DIR, argRefs);
  }

private:
  llvm::SmallVector<std::string> arguments;
};

class CompileToObjAction : public ActionTemplate<ActionData, ReturnCodeResult>,
                           public ClangExecutor {
public:
  CompileToObjAction(Compilation *copmilation, llvm::StringRef inputFileName,
                     llvm::StringRef outputFileName)
      : Base(copmilation), inputFileName(inputFileName),
        outputFileName(outputFileName) {}

  static llvm::StringRef getNameStr() { return "Compile to Object"; }
  llvm::StringRef getActionName() const override { return getNameStr(); }

  std::unique_ptr<ActionData> execute(std::unique_ptr<ActionData>) override {
    addArgument("-c");
    addArgument("-o", outputFileName);
    addArgument(inputFileName);
    addArgument("--target", "riscv64-unknown-linux-gnu");

    int returnCode = ClangExecutor::execute();
    if (returnCode != 0)
      return std::make_unique<ReturnCodeResult>(utils::LogicalResult::failure(),
                                                returnCode);

    return std::make_unique<ReturnCodeResult>(utils::LogicalResult::success(),
                                              returnCode);
  }

private:
  llvm::StringRef inputFileName;
  llvm::StringRef outputFileName;
};

class CompileToExeAction : public ActionTemplate<ActionData, ReturnCodeResult>,
                           public ClangExecutor {
public:
  CompileToExeAction(Compilation *compilation,
                     llvm::ArrayRef<llvm::StringRef> inputFileNames,
                     llvm::StringRef outputFileName)
      : Base(compilation), inputFileNames(inputFileNames),
        outputFileName(outputFileName) {}

  static llvm::StringRef getNameStr() { return "Compile to Executable"; }
  llvm::StringRef getActionName() const override { return getNameStr(); }

  std::unique_ptr<ActionData> execute(std::unique_ptr<ActionData>) override {
    addArgument("-o", outputFileName);
    for (auto inputFileName : inputFileNames) {
      addArgument(inputFileName);
    }

    addArgument("--target", "riscv64-unknown-linux-gnu");

    int returnCode = ClangExecutor::execute();
    if (returnCode != 0)
      return std::make_unique<ReturnCodeResult>(utils::LogicalResult::failure(),
                                                returnCode);
    return std::make_unique<ReturnCodeResult>(utils::LogicalResult::success(),
                                              returnCode);
  }

private:
  llvm::SmallVector<llvm::StringRef> inputFileNames;
  llvm::StringRef outputFileName;
};

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
                         llvm::StringRef inputSource,
                         llvm::SourceMgr &sourceMgr)
    : opt(opt), inputFileName(inputFileName), outputFileName(outputFileName),
      inputSource(inputSource), sourceMgr(sourceMgr) {}

int Compilation::compile() {
  Invocation mainInvocation;

  if (opt.inputFormat == InputFormat::C) {
    mainInvocation.addAction<ParseCAction>(this); // IR to Assembly
  } else if (opt.inputFormat == InputFormat::KeccIR) {
    mainInvocation.addAction<ParseIRAction>(this); // KeccIR source to IR
  }

  if (opt.optLevel == OptLevel::O1) {
    mainInvocation.addAction<RegisterPassesAction>(this,
                                                   translate::addO1Passes);
  }

  if (optPipeline)
    mainInvocation.addAction<RegisterPassesAction>(this, optPipeline);

  mainInvocation.addAction<RunPassesAction>(this);

  if (opt.outputFormat == OutputFormat::Assembly) {
    mainInvocation.addAction<TranslateIRAction>(this);
  }

  llvm::SmallVector<char> outputFileBuffer;
  if (opt.printStdOut) {
    mainInvocation.addAction<PrintAction>(this, llvm::outs());
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
        this,
        llvm::StringRef(outputFileBuffer.data(), outputFileBuffer.size()));
  }

  if (opt.outputFormat == OutputFormat::Object) {
    mainInvocation.addAction<CompileToObjAction>(
        this, llvm::StringRef(outputFileBuffer.data(), outputFileBuffer.size()),
        outputFileName);
  } else if (opt.outputFormat == OutputFormat::Executable) {
    mainInvocation.addAction<CompileToExeAction>(
        this,
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

} // namespace kecc
