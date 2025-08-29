#include "kecc/asm/AsmBuilder.h"
#include "kecc/asm/Register.h"
#include "kecc/translate/IRTranslater.h"
#include "kecc/translate/LiveRangeAnalyses.h"
#include "kecc/utils/LogicalResult.h"

namespace kecc {

utils::LogicalResult
translateCallLikeInstruction(as::AsmBuilder &builder, TranslationRule *rule,
                             FunctionTranslater &translater,
                             ir::InstructionStorage *inst) {
  assert(rule->restoreActively() && "Call instructions must restore actively");

  auto *module = translater.getModule();
  CallLivenessAnalysis *callLiveness =
      module->getAnalysis<CallLivenessAnalysis>();
  if (!callLiveness) {
    auto callLivenessAnalysis = CallLivenessAnalysis::create(module);
    module->insertAnalysis(std::move(callLivenessAnalysis));
    callLiveness = module->getAnalysis<CallLivenessAnalysis>();
  }

  RegisterAllocation *regAlloc = translater.getRegisterAllocation();
  LiveRangeAnalysis *liveRangeAnalysis = translater.getLiveRangeAnalysis();

  llvm::DenseSet<LiveRange> liveIn = callLiveness->getLiveIn(inst);
  llvm::SmallVector<std::pair<as::Register, as::DataSize>, 16> toSave;
  for (LiveRange lr : liveIn) {
    auto reg = regAlloc->getRegister(translater.getFunction(), lr);
    if (reg.isCallerSaved()) {
      auto dataSize = getDataSize(
          liveRangeAnalysis->getLiveRangeType(translater.getFunction(), lr));
      toSave.emplace_back(reg, dataSize);
    }
  }

  llvm::sort(toSave, [](const auto &a, const auto &b) {
    as::Register regA = a.first;
    as::Register regB = b.first;
    as::CommonRegisterLess less;
    return less(regA, regB);
  });

  auto memoryStackPointers =
      translater.saveCallerSavedRegisters(builder, toSave);

  auto result = rule->translate(builder, translater, inst);
  if (!result.succeeded())
    return result;

  translater.loadCallerSavedRegisters(builder, memoryStackPointers, toSave);

  return utils::LogicalResult::success();
}

utils::LogicalResult translateInstruction(as::AsmBuilder &builder,
                                          FunctionTranslater &translater,
                                          ir::InstructionStorage *inst) {
  auto *rule = translater.getTranslateContext()->getTranslateRuleSet()->getRule(
      inst->getAbstractInstruction()->getId());
  if (!rule) {
    inst->getContext()->diag().report(
        inst->getRange(), llvm::SourceMgr::DK_Error,
        std::format("No translation rule for instruction"));
    return utils::LogicalResult::error();
  }

  if (!rule->restoreActively()) {
    for (const ir::Operand &operand : inst->getOperands()) {
      translater.restoreOperand(builder, &operand);
    }
  }

  if (rule->callFunction()) {
    auto result = translateCallLikeInstruction(builder, rule, translater, inst);
    if (!result.succeeded())
      return result;
  } else {
    auto result = rule->translate(builder, translater, inst);
    if (!result.succeeded())
      return result;
  }

  for (ir::Value result : inst->getResults()) {
    auto liveRange = translater.getLiveRangeAnalysis()->getLiveRange(
        translater.getFunction(), result);
    if (translater.isSpilled(liveRange)) {
      translater.spillRegister(builder, liveRange);
    }
  }

  return utils::LogicalResult::success();
}

as::Block *translateBlock(as::AsmBuilder &builder,
                          FunctionTranslater &translater, ir::Block *block) {
  auto newBlock = translater.createBlock(block);
  for (auto I = block->tempBegin(), E = block->end(); I != E; ++I) {
    ir::InstructionStorage *inst = *I;
    auto result = translateInstruction(builder, translater, inst);
    if (!result.succeeded()) {
      delete newBlock;
      return nullptr;
    }
  }

  return newBlock;
}

} // namespace kecc
