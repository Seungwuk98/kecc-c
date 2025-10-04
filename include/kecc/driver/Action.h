#ifndef KECC_DRIVER_ACTION_H
#define KECC_DRIVER_ACTION_H

#include "kecc/asm/AsmInstruction.h"
#include "kecc/c/Diag.h"
#include "kecc/c/IRGenerator.h"
#include "kecc/c/ParseAST.h"
#include "kecc/driver/DriverConfig.h"
#include "kecc/driver/DriverConfig.h.in"
#include "kecc/ir/Context.h"
#include "kecc/ir/IR.h"
#include "kecc/ir/IRAnalyses.h"
#include "kecc/ir/Instruction.h"
#include "kecc/ir/Interpreter.h"
#include "kecc/ir/Pass.h"
#include "kecc/parser/Lexer.h"
#include "kecc/parser/Parser.h"
#include "kecc/translate/IRTranslater.h"
#include "kecc/utils/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Program.h"
#include <format>
#include <memory>

namespace kecc {

class Compilation;

class ActionData : public utils::PointerCastBase<ActionData> {
public:
  ActionData(utils::LogicalResult result);
  virtual ~ActionData() = default;

  TypeID getTypeID() const { return typeId; }
  utils::LogicalResult getLogicalResult() const { return result; }

  static llvm::StringRef getNameStr() {
    return "Anonymous Action Result or Argument";
  }
  virtual llvm::StringRef getName() const { return getNameStr(); }

  void setResult(utils::LogicalResult res) { result = res; }

protected:
  ActionData(utils::LogicalResult result, TypeID typeId)
      : result(result), typeId(typeId) {}

private:
  utils::LogicalResult result;
  TypeID typeId;
};

template <typename ConcreteType> struct ActionDataTemplate : public ActionData {
  using Base = ActionDataTemplate<ConcreteType>;

  ActionDataTemplate(utils::LogicalResult result)
      : ActionData(result, TypeID::get<ConcreteType>()) {}

  static bool classof(const ActionData *result) {
    return result->getTypeID() == TypeID::get<ConcreteType>();
  }
};

class Action {
public:
  Action() = default;

  virtual ~Action() = default;
  virtual llvm::StringRef getActionName() const { return "Anonymous Action"; };

  virtual void preExecute(ActionData *arg) {}
  virtual void postExecute(ActionData *result) {}

  virtual std::unique_ptr<ActionData>
  execute(std::unique_ptr<ActionData> arg) = 0;

  virtual llvm::StringRef getDescription() const { return ""; }
};

class CompilationAction : public Action {
public:
  CompilationAction(Compilation *compilation) : compilation(compilation) {}

  Compilation *getCompilation() const { return compilation; }
  virtual llvm::StringRef getActionName() const override {
    return "Compilation Action";
  }

private:
  Compilation *compilation;
};

template <typename ParentAction, typename ArgType, typename ResultType,
          bool Enable = true>
class ActionTemplate;

template <typename ParentAction, typename ResultType>
class ActionTemplate<ParentAction, ActionData, ResultType>
    : public ParentAction {
public:
  using Base = ActionTemplate<ParentAction, ActionData, ResultType>;
  using ParentAction::ParentAction;
};

template <typename ParentAction, typename ArgType, typename ResultType>
class ActionTemplate<ParentAction, ArgType, ResultType> : public ParentAction {
public:
  using Base = ActionTemplate<ParentAction, ArgType, ResultType>;
  using ParentAction::ParentAction;

  llvm::StringRef getActionName() const override {
    return ParentAction::getActionName();
  }

  void preExecute(ActionData *arg) override final {
    if (!arg->isa<ArgType>()) {
      llvm::errs() << "Action " << getActionName()
                   << " expected argument of type " << ArgType::getNameStr()
                   << ", but got " << arg->getName() << "\n";
      return;
    }
    preExecute(arg->cast<ArgType>());
  }

  virtual void preExecute(ArgType *arg) {}

  void postExecute(ActionData *result) override final {
    if (!result->isa<ResultType>()) {
      llvm::errs() << "Action " << getActionName()
                   << " expected result of type " << ResultType::getNameStr()
                   << ", but got " << result->getName() << "\n";
      return;
    }
    postExecute(result->cast<ResultType>());
  }

  virtual void postExecute(ResultType *result) {}

  std::unique_ptr<ActionData>
  execute(std::unique_ptr<ActionData> arg) override final {
    if (!arg->isa<ArgType>()) {
      llvm::errs() << "Action " << getActionName()
                   << " expected argument of type " << ArgType::getNameStr()
                   << ", but got " << arg->getName() << "\n";
      return nullptr;
    }
    std::unique_ptr<ArgType> castedArg(arg.release()->cast<ArgType>());
    return execute(std::move(castedArg));
  }

  virtual std::unique_ptr<ResultType> execute(std::unique_ptr<ArgType> arg) = 0;
};

class Invocation {
public:
  void addAction(std::unique_ptr<Action> action) {
    actions.emplace_back(std::move(action));
  }

  template <typename ActionType, typename... Args>
  void addAction(Args &&...args) {
    auto action = std::make_unique<ActionType>(std::forward<Args>(args)...);
    actions.emplace_back(std::move(action));
  }

  llvm::StringRef printAllSchedule() const;

  std::unique_ptr<ActionData> executeAll();

private:
  std::vector<std::unique_ptr<Action>> actions;
};

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

class ParseCAction
    : public ActionTemplate<CompilationAction, ActionData, OptArg> {
public:
  ParseCAction(Compilation *compilation, bool ignoreWarnings = false)
      : Base(compilation), ignoreWarnings(ignoreWarnings) {}

  static llvm::StringRef getNameStr() { return "Parse C source"; }
  llvm::StringRef getActionName() const override { return getNameStr(); }

  std::unique_ptr<ActionData> execute(std::unique_ptr<ActionData>) override;

private:
  bool ignoreWarnings;
};

class ParseIRAction
    : public ActionTemplate<CompilationAction, ActionData, OptArg> {
public:
  ParseIRAction(Compilation *compilation) : Base(compilation) {}

  static llvm::StringRef getNameStr() { return "Parse IR source"; }
  llvm::StringRef getActionName() const override { return getNameStr(); }

  std::unique_ptr<ActionData> execute(std::unique_ptr<ActionData>) override;
};

class InterpretResult : public ActionDataTemplate<InterpretResult> {
public:
  InterpretResult(
      utils::LogicalResult result,
      llvm::SmallVector<std::unique_ptr<ir::VRegister>> returnValues)
      : Base(result), returnValues(std::move(returnValues)) {}

  static llvm::StringRef getNameStr() { return "Interpret Result"; }
  llvm::StringRef getName() const override { return getNameStr(); }

  llvm::SmallVectorImpl<std::unique_ptr<ir::VRegister>> &getReturnValues() {
    return returnValues;
  }

private:
  llvm::SmallVector<std::unique_ptr<ir::VRegister>> returnValues;
};

class IRInterpretAction
    : public ActionTemplate<Action, OptArg, InterpretResult> {
public:
  IRInterpretAction(llvm::SmallVector<std::unique_ptr<ir::VRegister>> args,
                    llvm::StringRef entryFunction)
      : arguments(std::move(args)), entryFunctionName(entryFunction) {
    assert(entryFunction != "main");
  }

  IRInterpretAction(llvm::ArrayRef<llvm::StringRef> args)
      : arguments(args), entryFunctionName("main") {}

  static llvm::StringRef getNameStr() { return "Interpret IR"; }
  llvm::StringRef getActionName() const override { return getNameStr(); }

  std::unique_ptr<InterpretResult>
  execute(std::unique_ptr<OptArg> arg) override;

private:
  std::variant<llvm::SmallVector<std::unique_ptr<ir::VRegister>>,
               llvm::ArrayRef<llvm::StringRef>>
      arguments;
  llvm::StringRef entryFunctionName;
};

class RegisterPassesAction : public ActionTemplate<Action, OptArg, OptArg> {
public:
  RegisterPassesAction(llvm::ArrayRef<ir::Pass *> passes) : passes(passes) {}

  RegisterPassesAction(std::function<void(ir::PassManager &)> passRegisterFunc)
      : passes(std::move(passRegisterFunc)) {}

  static llvm::StringRef getNameStr() { return "Register Passes"; }
  llvm::StringRef getActionName() const override { return getNameStr(); }

  std::unique_ptr<OptArg> execute(std::unique_ptr<OptArg> arg) override;

private:
  std::variant<llvm::ArrayRef<ir::Pass *>,
               std::function<void(ir::PassManager &)>>
      passes;
};

class RunPassesAction : public ActionTemplate<Action, OptArg, OptArg> {
public:
  RunPassesAction() : Base() {}

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

class TranslateIRAction
    : public ActionTemplate<CompilationAction, OptArg, AsmArg> {
public:
  TranslateIRAction(Compilation *compilation) : Base(compilation) {}

  static llvm::StringRef getNameStr() { return "Translate to Assembly"; }
  llvm::StringRef getActionName() const override { return getNameStr(); }

  void preExecute(OptArg *) override;

  std::unique_ptr<AsmArg> execute(std::unique_ptr<OptArg> arg) override;
};

class OutputAction : public Action {
public:
  OutputAction(llvm::StringRef output) : outputFileName(output) {}

  static llvm::StringRef getNameStr() { return "Output Assembly"; }
  llvm::StringRef getActionName() const override { return getNameStr(); }

  std::unique_ptr<ActionData>
  execute(std::unique_ptr<ActionData> arg) override {
    std::error_code ec;
    llvm::raw_fd_ostream os(outputFileName, ec);
    if (ec) {
      llvm::errs() << "Error opening output file " << outputFileName << ": "
                   << ec.message() << "\n";
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
    return std::move(arg);
  }

private:
  std::string outputFileName;
};

class PrintAction : public Action {
public:
  PrintAction(llvm::raw_ostream &os) : os(&os) {}

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
    return std::move(arg);
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

  void addArguments(llvm::ArrayRef<llvm::StringRef> args) {
    for (auto arg : args) {
      addArgument(arg);
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

class CompileToObjAction
    : public ActionTemplate<Action, ActionData, ReturnCodeResult>,
      public ClangExecutor {
public:
  CompileToObjAction(llvm::StringRef inputFileName,
                     llvm::StringRef outputFileName,
                     llvm::StringRef triple = "riscv64-unknown-linux-gnu")
      : inputFileName(inputFileName), outputFileName(outputFileName),
        triple(triple) {}

  static llvm::StringRef getNameStr() { return "Compile to Object"; }
  llvm::StringRef getActionName() const override { return getNameStr(); }

  std::unique_ptr<ActionData> execute(std::unique_ptr<ActionData>) override {
    addArgument("-c");
    addArgument("-o", outputFileName);
    addArgument(inputFileName);
    addArgument("--target", triple);

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
  std::string triple;
};

class CompileToExeAction
    : public ActionTemplate<Action, ActionData, ReturnCodeResult>,
      public ClangExecutor {
public:
  CompileToExeAction(llvm::ArrayRef<llvm::StringRef> inputFileNames,
                     llvm::StringRef outputFileName,
                     llvm::ArrayRef<llvm::StringRef> extraArgs = {},
                     llvm::StringRef triple = "riscv64-unknown-linux-gnu")
      : inputFileNames(inputFileNames), outputFileName(outputFileName),
        extraArgs(extraArgs), triple(triple) {}

  static llvm::StringRef getNameStr() { return "Compile to Executable"; }
  llvm::StringRef getActionName() const override { return getNameStr(); }

  std::unique_ptr<ActionData> execute(std::unique_ptr<ActionData>) override {
    addArgument("-o");
    addArgument(outputFileName);
    for (auto inputFileName : inputFileNames) {
      addArgument(inputFileName);
    }

    addArgument("--target", triple);
    addArgument("-fuse-ld", "lld");
    if (!extraArgs.empty())
      addArguments(extraArgs);

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
  llvm::SmallVector<llvm::StringRef> extraArgs;
  std::string triple;
};

class CompileToExeGccAction
    : public ActionTemplate<Action, ActionData, ReturnCodeResult> {
public:
  CompileToExeGccAction(llvm::ArrayRef<llvm::StringRef> inputFileNames,
                        llvm::StringRef outputFileName,
                        llvm::ArrayRef<llvm::StringRef> extraArgs = {},
                        llvm::StringRef gccPath = GCC_DIR)
      : inputFileNames(inputFileNames), outputFileName(outputFileName),
        extraArgs(extraArgs), gccPath(gccPath) {}

  static llvm::StringRef getNameStr() { return "Compile to Executable by GCC"; }
  llvm::StringRef getActionName() const override { return getNameStr(); }

  std::unique_ptr<ActionData> execute(std::unique_ptr<ActionData>) override {
    llvm::SmallVector<llvm::StringRef> gccArgs{gccPath, "-o", outputFileName};
    for (auto inputFileName : inputFileNames) {
      gccArgs.push_back(inputFileName);
    }
    if (!extraArgs.empty())
      gccArgs.append(extraArgs.begin(), extraArgs.end());

    int returnCode = llvm::sys::ExecuteAndWait(gccPath, gccArgs);
    if (returnCode != 0)
      return std::make_unique<ReturnCodeResult>(utils::LogicalResult::failure(),
                                                returnCode);
    return std::make_unique<ReturnCodeResult>(utils::LogicalResult::success(),
                                              returnCode);
  }

private:
  llvm::SmallVector<llvm::StringRef> inputFileNames;
  llvm::StringRef outputFileName;
  llvm::SmallVector<llvm::StringRef> extraArgs;
  llvm::StringRef gccPath;
};

class ExecuteBinAction
    : public ActionTemplate<Action, ActionData, ReturnCodeResult> {
public:
  ExecuteBinAction(llvm::StringRef binPath,
                   llvm::ArrayRef<llvm::StringRef> args = {},
                   size_t timeout = 0)
      : binPath(binPath), args(args), timeout(timeout) {}

  static llvm::StringRef getNameStr() { return "Execute Binary"; }
  llvm::StringRef getActionName() const override { return getNameStr(); }

  std::unique_ptr<ActionData> execute(std::unique_ptr<ActionData>) override {
    llvm::SmallVector<llvm::StringRef> execArgs{binPath};
    if (!args.empty())
      execArgs.append(args.begin(), args.end());
    int returnCode = llvm::sys::ExecuteAndWait(binPath, execArgs, std::nullopt,
                                               std::nullopt, timeout);
    if (returnCode < 0)
      return std::make_unique<ReturnCodeResult>(utils::LogicalResult::failure(),
                                                returnCode);
    return std::make_unique<ReturnCodeResult>(utils::LogicalResult::success(),
                                              returnCode);
  }

private:
  llvm::StringRef binPath;
  llvm::SmallVector<llvm::StringRef> args;
  size_t timeout;
};

class ExecuteExeByQemuAction
    : public ActionTemplate<Action, ActionData, ReturnCodeResult> {
public:
  ExecuteExeByQemuAction(llvm::StringRef qemuPath, llvm::StringRef exePath,
                         llvm::ArrayRef<llvm::StringRef> args = {},
                         size_t timeout = 0)
      : qemuPath(qemuPath), exePath(exePath), timeout(timeout) {}

  static llvm::StringRef getNameStr() { return "Execute Executable by QEMU"; }
  llvm::StringRef getActionName() const override { return getNameStr(); }

  std::unique_ptr<ActionData> execute(std::unique_ptr<ActionData>) override {
    llvm::SmallVector<llvm::StringRef> qemuArgs{qemuPath, exePath};
    if (!args.empty())
      qemuArgs.append(args.begin(), args.end());
    int returnCode = llvm::sys::ExecuteAndWait(qemuPath, qemuArgs, std::nullopt,
                                               std::nullopt, timeout);
    if (returnCode < 0)
      return std::make_unique<ReturnCodeResult>(utils::LogicalResult::failure(),
                                                returnCode);
    return std::make_unique<ReturnCodeResult>(utils::LogicalResult::success(),
                                              returnCode);
  }

private:
  llvm::StringRef qemuPath;
  llvm::StringRef exePath;
  llvm::SmallVector<llvm::StringRef> args;
  size_t timeout;
};

} // namespace kecc

#endif // KECC_DRIVER_ACTION_H
