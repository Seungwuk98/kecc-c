#include "kecc/translate/RegisterAllocation.h"
#include "kecc/ir/IR.h"
#include "kecc/ir/IRInstructions.h"
#include "kecc/ir/IRTypes.h"
#include "kecc/ir/Instruction.h"
#include "kecc/ir/WalkSupport.h"
#include "kecc/translate/InterferenceGraph.h"
#include "kecc/translate/SpillAnalysis.h"
#include "kecc/translate/TranslateContext.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"

namespace kecc {

RegisterAllocation::RegisterAllocation(ir::Module *module,
                                       TranslateContext *translateContext)
    : module(module), translateContext(translateContext) {
  spillAnalysis = module->getAnalysis<SpillAnalysis>();
  liveRangeAnalysis = module->getAnalysis<LiveRangeAnalysis>();

  if (module->getIR()->hasFunctionDefinitions()) {
    assert(spillAnalysis && liveRangeAnalysis &&
           "SpillAnalysis and LiveRangeAnalysis must be available");

    allocateRegisters();
  }
}

as::Register RegisterAllocation::getRegister(ir::Value value) {
  LiveRange liveRange = liveRangeAnalysis->getLiveRange(value);
  ir::Function *function =
      value.getInstruction()->getParentBlock()->getParentFunction();
  return getRegister(function, liveRange);
}

as::Register RegisterAllocation::getRegister(ir::Function *function,
                                             LiveRange liveRange) {
  if (auto it = liveRangeToRegisterMap[function].find(liveRange);
      it != liveRangeToRegisterMap[function].end()) {
    return it->second;
  }
  InterferenceGraph *interferenceGraph;
  as::RegisterType regType;
  if (liveRange.getType().isa<ir::FloatT>()) {
    interferenceGraph = spillAnalysis->getInterferenceGraph(
        function, as::RegisterType::FloatingPoint);
    regType = as::RegisterType::FloatingPoint;
  } else {
    interferenceGraph = spillAnalysis->getInterferenceGraph(
        function, as::RegisterType::Integer);
    regType = as::RegisterType::Integer;
  }

  GraphColoring *graphColoring = interferenceGraph->getGraphColoring();
  auto color = graphColoring->getColor(liveRange);

  as::Register reg = colorToRegisterMap[function].at({regType, color});
  liveRangeToRegisterMap[function].try_emplace(liveRange, reg);
  return reg;
}

void RegisterAllocation::allocateRegisters() {
  auto intRegisters =
      translateContext->getRegistersForAllocate(as::RegisterType::Integer);
  auto floatRegisters = translateContext->getRegistersForAllocate(
      as::RegisterType::FloatingPoint);

  llvm::for_each(intRegisters, [&](as::Register reg) {
    if (reg.isArg())
      intArgRegisters.emplace_back(reg);
  });
  llvm::for_each(floatRegisters, [&](as::Register reg) {
    if (reg.isArg())
      floatArgRegisters.emplace_back(reg);
  });

  llvm::sort(intArgRegisters, [](as::Register a, as::Register b) {
    return a.getABIIndex() < b.getABIIndex();
  });
  llvm::sort(floatArgRegisters, [](as::Register a, as::Register b) {
    return a.getABIIndex() < b.getABIIndex();
  });

  for (ir::Function *function : *module->getIR()) {
    auto intInterferenceGraph = spillAnalysis->getInterferenceGraph(
        function, as::RegisterType::Integer);
    auto floatInterferenceGraph = spillAnalysis->getInterferenceGraph(
        function, as::RegisterType::FloatingPoint);

    allocateRegistersForFunction(function, intInterferenceGraph,
                                 floatInterferenceGraph);
  }
}

void RegisterAllocation::allocateRegistersForFunction(
    ir::Function *function, InterferenceGraph *intInterferenceGraph,
    InterferenceGraph *floatInterferenceGraph) {

  GraphColoring *intGraphColoring = intInterferenceGraph->coloring();
  GraphColoring *floatGraphColoring = floatInterferenceGraph->coloring();

  using Color = GraphColoring::Color;

  llvm::DenseSet<as::Register> usedRegisters;
  llvm::DenseMap<Color, as::Register> intColorToRegister;
  llvm::DenseMap<Color, as::Register> floatColorToRegister;

  ir::Block *entryBlock = function->getEntryBlock();
  ir::FunctionT funcT = function->getFunctionType().cast<ir::FunctionT>();

  llvm::ArrayRef<as::Register> intRegisters =
      translateContext->getRegistersForAllocate(as::RegisterType::Integer);
  llvm::ArrayRef<as::Register> floatRegisters =
      translateContext->getRegistersForAllocate(
          as::RegisterType::FloatingPoint);

  // special case for register allocations
  // 1) function argument

  llvm::SmallVector<LiveRange, 8> intArgLiveRanges;
  llvm::SmallVector<LiveRange, 8> floatArgLiveRanges;

  for (auto I = entryBlock->begin();
       (*I)->getDefiningInst<ir::inst::FunctionArgument>(); ++I) {
    ir::inst::FunctionArgument argInst =
        (*I)->getDefiningInst<ir::inst::FunctionArgument>();

    auto liveRange = liveRangeAnalysis->getLiveRange(argInst);
    if (argInst.getType().isa<ir::FloatT>())
      floatArgLiveRanges.emplace_back(liveRange);
    else
      intArgLiveRanges.emplace_back(liveRange);
  }

  for (auto idx = 0u;
       idx < intArgLiveRanges.size() && idx < intArgRegisters.size(); ++idx) {
    LiveRange liveRange = intArgLiveRanges[idx];
    Color color = intGraphColoring->getColor(liveRange);
    as::Register reg = intArgRegisters[idx];
    auto [_, inserted] = intColorToRegister.try_emplace(color, reg);
    assert(inserted && "Argument register color already assigned");
    inserted = usedRegisters.insert(reg).second;
    assert(inserted && "register already used");
  }

  for (auto idx = 0u;
       idx < floatArgLiveRanges.size() && idx < floatArgRegisters.size();
       ++idx) {
    LiveRange liveRange = floatArgLiveRanges[idx];
    Color color = floatGraphColoring->getColor(liveRange);
    as::Register reg = floatArgRegisters[idx];
    auto [_, inserted] = floatColorToRegister.try_emplace(color, reg);
    assert(inserted && "Argument register color already assigned");
    inserted = usedRegisters.insert(reg).second;
    assert(inserted && "register already used");
  }

  // 2) function return value
  static llvm::SmallVector<as::Register> intReturnRegisters = {
      as::Register::a0(),
      as::Register::a1(),
  };

  static llvm::SmallVector<as::Register> floatReturnRegisters = {
      as::Register::fa0(),
      as::Register::fa1(),
  };

  function->walk([&](ir::InstructionStorage *inst) -> ir::WalkResult {
    ir::inst::Return retInst = inst->getDefiningInst<ir::inst::Return>();
    if (!retInst)
      return ir::WalkResult::advance();

    llvm::ArrayRef<ir::Operand> returnValues = retInst.getValues();
    llvm::SmallVector<LiveRange, 8> intReturnLiveRanges;
    llvm::SmallVector<LiveRange, 8> floatReturnLiveRanges;

    for (const ir::Operand &operand : returnValues) {
      LiveRange liveRange;
      if (spillAnalysis->getSpillInfo().restore.contains(&operand))
        liveRange = spillAnalysis->getSpillInfo().restore.at(&operand);
      else
        liveRange = liveRangeAnalysis->getLiveRange(operand);

      if (operand.getType().isa<ir::FloatT>())
        floatReturnLiveRanges.emplace_back(liveRange);
      else
        intReturnLiveRanges.emplace_back(liveRange);
    }

    assert(intReturnLiveRanges.size() <= intReturnRegisters.size() &&
           "Too many integer return values for available registers");
    assert(floatReturnLiveRanges.size() <= floatReturnRegisters.size() &&
           "Too many floating point return values for available registers");

    for (auto idx = 0u; idx < intReturnLiveRanges.size(); ++idx) {
      LiveRange liveRange = intReturnLiveRanges[idx];
      Color color = intGraphColoring->getColor(liveRange);
      if (intColorToRegister.contains(color))
        continue;

      as::Register reg = intReturnRegisters[idx];
      if (usedRegisters.contains(reg))
        continue;
      intColorToRegister.try_emplace(color, reg);
      usedRegisters.insert(reg);
    }

    for (auto idx = 0u; idx < floatReturnLiveRanges.size(); ++idx) {
      LiveRange liveRange = floatReturnLiveRanges[idx];
      Color color = floatGraphColoring->getColor(liveRange);
      if (floatColorToRegister.contains(color))
        continue;
      as::Register reg = floatReturnRegisters[idx];
      if (usedRegisters.contains(reg))
        continue;
      floatColorToRegister.try_emplace(color, reg);
      usedRegisters.insert(reg);
    }
    return ir::WalkResult::advance();
  });

  auto intRegIdx = 0u;
  for (Color color = 0u; color < intGraphColoring->getNumColors(); ++color) {
    if (intColorToRegister.contains(color))
      continue;

    for (; usedRegisters.contains(intRegisters[intRegIdx]); ++intRegIdx)
      ;
    as::Register reg = intRegisters[intRegIdx++];
    intColorToRegister.try_emplace(color, reg);
    usedRegisters.insert(reg);
  }

  auto floatRegIdx = 0u;
  for (Color color = 0u; color < floatGraphColoring->getNumColors(); ++color) {
    if (floatColorToRegister.contains(color))
      continue;

    for (; usedRegisters.contains(floatRegisters[floatRegIdx]); ++floatRegIdx)
      ;
    as::Register reg = floatRegisters[floatRegIdx++];
    floatColorToRegister.try_emplace(color, reg);
    usedRegisters.insert(reg);
  }

  for (const auto &[color, reg] : intColorToRegister) {
    auto &colorMap = colorToRegisterMap[function];
    auto [_, inserted] = colorMap.try_emplace(
        std::make_pair(as::RegisterType::Integer, color), reg);
    assert(inserted && "Integer register color already assigned");
    (void)inserted;
  }

  for (const auto &[color, reg] : floatColorToRegister) {
    auto &colorMap = colorToRegisterMap[function];
    auto [_, inserted] = colorMap.try_emplace(
        std::make_pair(as::RegisterType::FloatingPoint, color), reg);
    assert(inserted && "Floating point register color already assigned");
    (void)inserted;
  }
}

} // namespace kecc
