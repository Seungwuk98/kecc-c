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

FunctionTranslater::FunctionTranslater(TranslateContext *context,
                                       ir::Module *module,
                                       ir::Function *function)
    : context(context), module(module), function(function) {}

as::Block *FunctionTranslater::createBlock(ir::Block *block) {
  auto newName = std::format(
      ".{}_L{}", block->getParentFunction()->getName().str(), block->getId());
  auto *newBlock = new as::Block(newName);
  return newBlock;
}

std::unique_ptr<as::Asm> IRTranslater::translate() {
  llvm::SmallVector<as::Section<as::Function> *> functions;

  for (ir::Function *function : *module->getIR()) {
    FunctionTranslater translater(context, module, function);

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
