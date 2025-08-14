#include "kecc/asm/Asm.h"
#include "kecc/ir/Context.h"
#include "kecc/ir/IRTransforms.h"
#include "kecc/ir/Instruction.h"
#include "kecc/ir/Module.h"
#include "kecc/ir/Pass.h"
#include "kecc/parser/Parser.h"
#include "kecc/translate/IRTranslater.h"
#include "kecc/translate/SpillAnalysis.h"
#include "kecc/translate/TranslateContext.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include <string>

namespace kecc::as {}

namespace kecc::cl {

enum Action {
  DumpIR,
  DumpInterferenceGraph,
  DumpLiveRangeWithSpill,
  DumpSpillCost,
  DumpRegisterAllocation,
  TranslateIR,
};

static llvm::cl::opt<std::string> input(llvm::cl::Positional,
                                        llvm::cl::init("-"),
                                        llvm::cl::desc("<input file>"));
static llvm::cl::opt<std::string> output("o", llvm::cl::init("-"),
                                         llvm::cl::desc("Specify output file"),
                                         llvm::cl::value_desc("filename"));

static llvm::cl::opt<Action>
    action("action", llvm::cl::desc("Action to perform:"),
           llvm::cl::init(TranslateIR),
           llvm::cl::values(
               clEnumValN(DumpIR, "dump-ir", "Dump IR to the output file"),
               clEnumValN(DumpInterferenceGraph, "dump-graph",
                          "Dump interference graph"),
               clEnumValN(DumpLiveRangeWithSpill, "dump-live-range-with-spill",
                          "Dump live range with spill"),
               clEnumValN(DumpSpillCost, "dump-spill-cost", "Dump spill cost"),
               clEnumValN(DumpRegisterAllocation, "dump-register-allocation",
                          "Dump register allocation"),
               clEnumValN(TranslateIR, "translate-ir", "Translate IR to ASM")));

static llvm::cl::opt<int> spillIterationsCount(
    "spill-iterations",
    llvm::cl::desc("Number of iterations for spilling. Default is infinite "
                   "until the program can be translated"),
    llvm::cl::init(-1), llvm::cl::value_desc("iterations"));

static llvm::cl::opt<std::string> registerForAllocation(
    "reg-for-alloc",
    llvm::cl::desc("Declare registers for register allocation"));

} // namespace kecc::cl

namespace kecc {

std::unique_ptr<ir::Module> parseAndBuildModule(ir::IRContext &context) {
  auto inputBufferOrErr = llvm::MemoryBuffer::getFileOrSTDIN(cl::input);
  if (inputBufferOrErr.getError()) {
    llvm::errs() << "read file error: " << inputBufferOrErr.getError().message()
                 << "\n";
    return nullptr;
  }

  auto index = context.getSourceMgr().AddNewSourceBuffer(
      std::move(inputBufferOrErr.get()), llvm::SMLoc());
  auto inputBuffRef = context.getSourceMgr().getMemoryBuffer(index);

  Lexer lexer(inputBuffRef->getBuffer(), &context);
  Parser parser(lexer, &context);

  std::unique_ptr<ir::Module> module = parser.parseAndBuildModule();

  if (context.diag().hasError()) {
    llvm::errs() << "Parsing failed with " << context.diag().getErrorCount()
                 << " errors.\n";
    return nullptr;
  }

  return module;
}

ir::PassResult runPasses(ir::Module *module) {
  ir::PassManager pm;

  pm.addPass<ir::OutlineConstantPass>();
  pm.addPass<ir::InlineCallPass>();
  pm.addPass<ir::CanonicalizeStruct>();
  pm.addPass<ir::OutlineMultipleResults>();
  pm.addPass<ir::CreateFunctionArgument>();
  pm.addPass<ir::CanonicalizeConstant>();

  return pm.run(module);
}

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

int keccTranslateMain() {
  ir::IRContext context;

  auto module = parseAndBuildModule(context);
  if (!module)
    return 1;

  auto passResult = runPasses(module.get());
  if (passResult.isFailure()) {
    llvm::errs() << "Passes failed\n";
    return 1;
  }

  if (cl::action == cl::DumpIR) {
    dump([&](llvm::raw_ostream &os) {
      ir::IRPrintContext printContext(os);
      module->getIR()->print(printContext);
    });
    return 0;
  }

  TranslateContext translateContext;

  if (cl::registerForAllocation.getNumOccurrences() > 0) {
    auto result =
        translateContext.setRegistersFromOption(cl::registerForAllocation);
    if (!result.succeeded()) {
      llvm::errs() << "Setting registers for allocation is failed\n";
      return 1;
    }
  } else {
    defaultRegisterSetup(&translateContext);
  }

  module->insertAnalysis(
      SpillAnalysis::create(module.get(), &translateContext));
  auto *spillAnalysis = module->getAnalysis<SpillAnalysis>();

  if (cl::spillIterationsCount >= 0) {
    spillAnalysis->trySpill(cl::spillIterationsCount);
  } else {
    spillAnalysis->spillFull();
  }

  if (cl::action == cl::DumpInterferenceGraph) {
    dump([&](llvm::raw_ostream &os) {
      spillAnalysis->dumpInterferenceGraph(os);
    });
    return 0;
  }

  if (cl::action == cl::DumpLiveRangeWithSpill) {
    auto liveRangeAnalysis = module->getAnalysis<LiveRangeAnalysis>();
    assert(liveRangeAnalysis && "LiveRangeAnalysis must not be null");

    dump([&](llvm::raw_ostream &os) {
      liveRangeAnalysis->dump(os, spillAnalysis->getSpillInfo());
    });
    return 0;
  }

  if (cl::action == cl::DumpSpillCost) {
    for (ir::Function *function : *module->getIR()) {
      dump([&](llvm::raw_ostream &os) {
        os << "Spill cost for function: @" << function->getName() << "\n";
        os << "For integer live ranges:\n";
        auto interfGraph = spillAnalysis->getInterferenceGraph(
            function, as::RegisterType::Integer);
        assert(interfGraph && "Interference graph must not be null");
        SpillCost spillCost(module.get(), function, interfGraph);
        spillCost.dump(os);

        os << "For floating-point live ranges:\n";
        interfGraph = spillAnalysis->getInterferenceGraph(
            function, as::RegisterType::FloatingPoint);
        assert(interfGraph && "Interference graph must not be null");
        spillCost = SpillCost(module.get(), function, interfGraph);
        spillCost.dump(os);
        os << '\n';
      });
    }
  }

  return 0;
}

} // namespace kecc

int main(int argc, const char **argv) {
  llvm::InitLLVM x(argc, argv);

  kecc::ir::registerPass<kecc::ir::CanonicalizeConstant>();
  kecc::ir::registerPass<kecc::ir::OutlineConstantPass>();
  kecc::ir::registerPass<kecc::ir::InlineCallPass>();
  kecc::ir::registerPass<kecc::ir::CanonicalizeStruct>();
  kecc::ir::registerPass<kecc::ir::OutlineMultipleResults>();
  kecc::ir::registerPass<kecc::ir::CreateFunctionArgument>();

  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "Kecc IR Translation Compiler\n");

  return kecc::keccTranslateMain();
}
