#include "kecc/driver/Action.h"
#include "kecc/asm/Register.h"
#include "kecc/driver/Compilation.h"
#include "kecc/ir/Instruction.h"

namespace kecc {

ActionData::ActionData(utils::LogicalResult result)
    : result(result), typeId(TypeID::get<ActionData>()) {}

std::unique_ptr<ActionData> Invocation::executeAll() {
  std::unique_ptr<ActionData> currentArg = nullptr;

  for (const std::unique_ptr<Action> &action : actions) {
    action->preExecute(currentArg.get());
    currentArg = action->execute(std::move(currentArg));
    if (!currentArg->getLogicalResult().succeeded())
      return currentArg;
    action->postExecute(currentArg.get());
  }

  return std::move(currentArg);
}

std::unique_ptr<ActionData>
ParseCAction::execute(std::unique_ptr<ActionData> input) {
  auto &compilation = *getCompilation();

  if (!compilation.getASTUnit()) {
    c::ParseAST parser(compilation.getInputSource(),
                       compilation.getInputFileName(), llvm::errs(),
                       ignoreWarnings);
    auto result = parser.parse();
    if (!result.succeeded())
      return std::make_unique<OptArg>(utils::LogicalResult::failure(), nullptr);

    clang::SourceManager &sm = parser.getASTUnit()->getSourceManager();
    llvm::StringRef sourceData = sm.getBufferData(sm.getMainFileID());

    auto newBuffer = llvm::MemoryBuffer::getMemBuffer(sourceData);
    compilation.getSourceMgr().AddNewSourceBuffer(std::move(newBuffer), {});
    compilation.setASTUnit(parser.releaseASTUnit());
  }

  clang::ASTUnit *astUnit = compilation.getASTUnit();

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

std::unique_ptr<ActionData>
ParseIRAction::execute(std::unique_ptr<ActionData> input) {
  auto &compilation = *getCompilation();

  Lexer lexer(compilation.getInputSource(), compilation.getIRContext());
  Parser parser(lexer, compilation.getIRContext());

  std::unique_ptr<ir::Module> module = parser.parseAndBuildModule();
  if (compilation.getIRContext()->diag().hasError())
    return std::make_unique<OptArg>(utils::LogicalResult::failure(), nullptr);

  return std::make_unique<OptArg>(utils::LogicalResult::success(),
                                  std::move(module));
}

std::unique_ptr<InterpretResult>
IRInterpretAction::execute(std::unique_ptr<OptArg> arg) {
  ir::Module *module = arg->getModule();
  ir::Interpreter interpreter(module);

  if (entryFunctionName.empty() || entryFunctionName == "main") {
    assert(std::holds_alternative<llvm::ArrayRef<llvm::StringRef>>(arguments) &&
           "Entry function is 'main', but arguments are not string refs");
    auto args = std::get<llvm::ArrayRef<llvm::StringRef>>(arguments);
    int returnCode = interpreter.callMain(args);
    llvm::SmallVector<std::unique_ptr<ir::VRegister>> returnValues;
    returnValues.emplace_back(std::make_unique<ir::VRegisterInt>(returnCode));
    return std::make_unique<InterpretResult>(utils::LogicalResult::success(),
                                             std::move(returnValues));
  } else {
    assert(std::holds_alternative<
               llvm::SmallVector<std::unique_ptr<ir::VRegister>>>(arguments) &&
           "Entry function is not 'main'");
    auto args = llvm::map_to_vector(
        std::get<llvm::SmallVector<std::unique_ptr<ir::VRegister>>>(arguments),
        [](std::unique_ptr<ir::VRegister> &reg) { return reg.get(); });
    auto returnValues = interpreter.call(entryFunctionName, args);
    return std::make_unique<InterpretResult>(utils::LogicalResult::success(),
                                             std::move(returnValues));
  }
}
std::unique_ptr<OptArg>
RegisterPassesAction::execute(std::unique_ptr<OptArg> arg) {
  auto &pm = arg->getPassManager();
  std::visit(
      [&]<typename T>(T &&passes) {
        using PassesType = std::decay_t<T>;
        if constexpr (std::is_same_v<PassesType, llvm::ArrayRef<ir::Pass *>>) {
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

void TranslateIRAction::preExecute(OptArg *) {
  if (!getCompilation()->getTranslateContext())
    getCompilation()->createTranslateContext();

  defaultRegisterSetup(getCompilation()->getTranslateContext());
  registerDefaultTranslationRules(getCompilation()->getTranslateContext());
}

std::unique_ptr<AsmArg>
TranslateIRAction::execute(std::unique_ptr<OptArg> arg) {
  TranslateContext *translateContext = getCompilation()->getTranslateContext();
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
} // namespace kecc
