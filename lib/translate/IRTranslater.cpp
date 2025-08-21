#include "kecc/translate/IRTranslater.h"
#include "kecc/asm/AsmBuilder.h"
#include "kecc/asm/Register.h"
#include "kecc/ir/IRAnalyses.h"
#include "kecc/translate/InterferenceGraph.h"
#include "kecc/translate/SpillAnalysis.h"
#include <format>
#include <queue>

namespace kecc {

void defaultRegisterSetup(TranslateContext *context) {
  static llvm::ArrayRef<as::Register> tempIntRegs = {
      as::Register::t0(),
      as::Register::t1(),
  };

  context->setTempRegisters(tempIntRegs);

  llvm::SmallVector<as::Register, 32> intRegs;
  auto tempIntRegsForAlloc = as::getIntTempRegisters();
  tempIntRegsForAlloc = tempIntRegsForAlloc.drop_front(tempIntRegs.size());
  intRegs.append(tempIntRegsForAlloc.begin(), tempIntRegsForAlloc.end());
  intRegs.append(as::getIntArgRegisters().begin(),
                 as::getIntArgRegisters().end());
  intRegs.append(as::getIntSavedRegisters().begin(),
                 as::getIntSavedRegisters().end());

  llvm::SmallVector<as::Register, 32> floatRegs;
  floatRegs.append(as::getFpTempRegisters().begin(),
                   as::getFpTempRegisters().end());
  floatRegs.append(as::getFpArgRegisters().begin(),
                   as::getFpArgRegisters().end());
  floatRegs.append(as::getFpSavedRegisters().begin(),
                   as::getFpSavedRegisters().end());

  context->setRegistersForAllocate(intRegs, floatRegs);
}

extern void translateInstruction(as::AsmBuilder &builder,
                                 FunctionTranslater &translater,
                                 ir::InstructionStorage *inst);

FunctionTranslater::FunctionTranslater(IRTranslater *translater,
                                       TranslateContext *context,
                                       ir::Module *module,
                                       ir::Function *function)
    : irTranslater(translater), context(context), module(module),
      function(function), regAlloc(module, context) {
  liveRangeAnalysis = module->getAnalysis<LiveRangeAnalysis>();
  livenessAnalysis = module->getAnalysis<LivenessAnalysis>();
  spillAnalysis = module->getAnalysis<SpillAnalysis>();

  assert(liveRangeAnalysis && livenessAnalysis && spillAnalysis &&
         "LiveRangeAnalysis, LivenessAnalysis, and SpillAnalysis must be "
         "available");
}

llvm::SmallVector<as::Register> FunctionTranslater::saveCallerSavedRegisters(
    as::AsmBuilder &builder,
    llvm::ArrayRef<std::pair<as::Register, as::DataSize>> datas) {
  size_t totalSize = 0u;

  llvm::SmallVector<StackPoint> stackPoint;
  stackPoint.reserve(datas.size());
  for (const auto &[reg, dataSize] : datas) {
    stackPoint.emplace_back(stack.callerSavedRegister(totalSize));
    totalSize += dataSize.getByteSize();
  }

  llvm::SmallVector<as::Register> anonRegs;
  anonRegs.reserve(datas.size());
  for (const auto &sp : stackPoint) {
    auto anon = as::Register::createAnonymousRegister(
        context->getAnonymousRegStorage(), as::RegisterType::Integer,
        as::CallingConvension::None,
        std::format("{}_call{}_{}", function->getName().str(),
                    getCurrCallIndex(), getCurrCallRegIndex()));

    incrementCallRegIndex();
    anonymousRegisterToSp.try_emplace(anon, sp);
    anonRegs.emplace_back(anon);
  }

  for (size_t i = 0; i < datas.size(); ++i) {
    const auto &[reg, dataSize] = datas[i];
    const auto &anonStackReg = anonRegs[i];

    storeData(builder, *this, anonStackReg, reg, dataSize, 0);
  }

  return anonRegs;
}

void FunctionTranslater::loadCallerSavedRegisters(
    as::AsmBuilder &builder, llvm::ArrayRef<as::Register> stackpointers,
    llvm::ArrayRef<std::pair<as::Register, as::DataSize>> datas) {
  assert(stackpointers.size() == datas.size() &&
         "Stack pointers and data sizes must match in size");

  for (size_t i = 0; i < stackpointers.size(); ++i) {
    const auto &sp = stackpointers[i];
    const auto &[reg, dataSize] = datas[i];

    loadData(builder, *this, reg, sp, dataSize, 0);
  }
}

std::string FunctionTranslater::getBlockName(ir::Block *block) {
  // Generate a unique name for the block based on its function and ID
  return std::format(".{}_L{}", block->getParentFunction()->getName().str(),
                     block->getId());
}

as::Block *FunctionTranslater::createBlock(ir::Block *block) {
  auto newName = getBlockName(block);
  auto *newBlock = new as::Block(newName);
  return newBlock;
}

std::unique_ptr<as::Asm> IRTranslater::translate() {
  llvm::SmallVector<as::Section<as::Function> *> functions;

  for (ir::Function *function : *module->getIR()) {
    FunctionTranslater translater(this, context, module, function);

    auto *asFunction = translater.translate();
    assert(asFunction && "Function translation failed");

    auto *section = new as::Section<as::Function>({}, asFunction);
    functions.emplace_back(section);
  }

  return nullptr;
}

as::Function *FunctionTranslater::translate() {
  llvm::SmallVector<as::Block *> blocks;
  blocks.reserve(function->getBlockCount() + 1);
  auto funcEntryBlock = new as::Block(function->getName());
  blocks.emplace_back(funcEntryBlock);

  as::AsmBuilder builder;
  for (ir::Block *block : *function) {
    auto *asBlock = createBlock(block);
    blocks.emplace_back(asBlock);

    // Translate instructions in the block
    for (ir::InstructionStorage *inst : *block) {
      builder.setInsertionPointLast(asBlock);
      translateInstruction(builder, *this, inst);
    }
  }

  return new as::Function(blocks);
}

} // namespace kecc
