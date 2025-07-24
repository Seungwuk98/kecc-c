#include "kecc/ir/Context.h"
#include "kecc/ir/IRAnalyses.h"
#include "kecc/ir/IRTransforms.h"
#include "kecc/ir/Instruction.h"
#include "kecc/ir/Pass.h"
#include "kecc/parser/Parser.h"
#include "kecc/utils/Diag.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/raw_ostream.h"

namespace kecc::cl {

enum Action {
  Parse,
  Opt,
  DumpAnalysis,
};

enum AnalysisKind {
  Dominance,
};

static llvm::cl::opt<string> input(llvm::cl::Positional, llvm::cl::init("-"),
                                   llvm::cl::desc("<input file>"));

static llvm::cl::opt<string> output("o", llvm::cl::init("-"),
                                    llvm::cl::desc("Specify output file"),
                                    llvm::cl::value_desc("filename"));

static llvm::cl::opt<Action>
    action("action", llvm::cl::desc("Action to perform:"), llvm::cl::init(Opt),
           llvm::cl::values(clEnumValN(Parse, "parse", "Parse the input file"),
                            clEnumValN(Opt, "opt", "Optimize the input file"),
                            clEnumValN(DumpAnalysis, "dump-analysis",
                                       "Dump analysis results")));

static llvm::cl::opt<AnalysisKind> dumpAnalysisKind(
    "dump-analysis",
    llvm::cl::values(clEnumValN(Dominance, "dominance",
                                "Dump dominance analysis results")),
    llvm::cl::desc("Dump analysis results to the specified file"));

static llvm::cl::opt<bool> debugInfo(
    "debug-info", llvm::cl::init(false),
    llvm::cl::desc(
        "Generate information for debugging compiler in the output file"));

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

} // namespace kecc::cl

namespace kecc {

void dump(llvm::function_ref<void(llvm::raw_ostream &)> fn) {
  if (cl::output == "-") {
    fn(llvm::outs());
  } else {
    std::error_code ec;
    llvm::raw_fd_ostream os(cl::output, ec);
    if (ec) {
      llvm::errs() << "Error opening output file: " << ec.message() << "\n";
      return;
    }
    fn(os);
  }
}

void dumpModule(ir::Module *module) {
  dump([&](llvm::raw_ostream &os) {
    ir::IRPrintContext printContext(os, cl::debugInfo
                                            ? ir::IRPrintContext::Debug
                                            : ir::IRPrintContext::Default);
    module->getIR()->print(printContext);
  });
}

int dumpAnalysis(ir::Module *module) {
  switch (cl::dumpAnalysisKind) {
  case cl::Dominance: {
    auto domAnalysis = ir::DominanceAnalysis::create(module);
    assert(domAnalysis && "Failed to create dominance analysis");
    dump([&](llvm::raw_ostream &os) { domAnalysis->dump(os); });
  }
  }
  return 0;
}

int keccMain() {
  ir::IRContext context;

  if (cl::action == cl::Opt && cl::action.getNumOccurrences() == 0 &&
      cl::dumpAnalysisKind.getNumOccurrences()) {
    cl::action = cl::DumpAnalysis;
  }

  if (cl::action == cl::DumpAnalysis) {
    if (cl::dumpAnalysisKind.getNumOccurrences() == 0) {
      cl::dumpAnalysisKind.error("dump analysis requires a analysis name");
      return 1;
    }
  }

  auto inputBufferOrErr = llvm::MemoryBuffer::getFileOrSTDIN(cl::input);
  if (inputBufferOrErr.getError()) {
    llvm::errs() << "read file error: " << inputBufferOrErr.getError().message()
                 << '\n';
    return 1;
  }

  auto index = context.getSourceMgr().AddNewSourceBuffer(
      std::move(inputBufferOrErr.get()), {});
  auto inputBufferRef = context.getSourceMgr().getMemoryBuffer(index);

  Lexer lexer(inputBufferRef->getBuffer(), &context);
  Parser parser(lexer, &context);

  std::unique_ptr<ir::Module> module = parser.parseAndBuildModule();

  if (context.diag().hasError()) {
    llvm::errs() << "Parsing failed with " << context.diag().getErrorCount()
                 << " errors.\n";
    return 1;
  }

  if (cl::action == cl::Parse) {
    dumpModule(module.get());
    return 0;
  }

  if (cl::action == cl::DumpAnalysis) {
    return dumpAnalysis(module.get());
  }

  if (cl::action == cl::Opt) {
    ir::PassManager pm;

    cl::pmOption->passCallback(pm);
    auto result = pm.run(module.get());
    if (result.isFailure())
      return 1;
    dumpModule(module.get());
    return 0;
  }

  llvm_unreachable("Unknown action specified");
}

} // namespace kecc

int main(int argc, const char **argv) {
  llvm::InitLLVM x(argc, argv);

  kecc::ir::registerPass<kecc::ir::CanonicalizeConstant>();
  kecc::ir::registerPass<kecc::ir::Mem2Reg>();
  kecc::ir::registerPass<kecc::ir::GVN>();
  kecc::ir::registerPass<kecc::ir::DeadCode>();
  kecc::ir::registerPass<kecc::ir::OutlineConstant>();
  kecc::ir::registerPass<kecc::ir::InstructionFold>();
  kecc::ir::registerSimplifyCFGPass();
  kecc::cl::registerPMOption();
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "Kecc IR Optimization Compiler\n");
  return kecc::keccMain();
}
