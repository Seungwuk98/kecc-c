#ifndef KECC_TRANSLATER_H
#define KECC_TRANSLATER_H

#include "kecc/asm/Asm.h"
#include "kecc/asm/AsmBuilder.h"
#include "kecc/asm/AsmInstruction.h"
#include "kecc/ir/IRInstructions.h"
#include "kecc/ir/Module.h"
#include "kecc/ir/Value.h"
#include "kecc/translate/FunctionStack.h"
#include "kecc/translate/InterferenceGraph.h"
#include "kecc/translate/LiveRangeAnalyses.h"
#include "kecc/translate/RegisterAllocation.h"
#include "kecc/translate/TranslateContext.h"
#include "kecc/utils/LogicalResult.h"
#include <format>

namespace kecc {

class TranslationRule;

void defaultRegisterSetup(TranslateContext *context);

class IRTranslater {
public:
  IRTranslater(TranslateContext *context, ir::Module *module)
      : context(context), module(module) {}

  std::unique_ptr<as::Asm> translate();

  TranslateContext *getContext() const { return context; }
  ir::Module *getModule() const { return module; }

private:
  TranslateContext *context;
  ir::Module *module;
};

class FunctionTranslater {
public:
  FunctionTranslater(IRTranslater *translater, TranslateContext *context,
                     ir::Module *module, ir::Function *function);

  static std::string getBlockName(ir::Block *block);
  as::Block *createBlock(ir::Block *block);

  as::Function *translate();

  TranslateContext *getTranslateContext() const { return context; }

  as::Register getRegister(LiveRange liveRange);
  as::Register getRegister(ir::Value value);
  as::Register getOperandRegister(const ir::Operand *operand);

  std::pair<llvm::StringRef, as::DataSize>
  getConstantLabel(ir::ConstantAttr constant);
  std::pair<llvm::StringRef, as::DataSize> getConstantLabel(std::int64_t value);

  ir::Module *getModule() const { return module; }

  as::Register restoreOperand(as::AsmBuilder &builder,
                              const ir::Operand *operand);
  void spillRegister(as::AsmBuilder &builder, LiveRange liveRange);
  bool isSpilled(LiveRange liveRange) const;

  void writeFunctionEnd(as::AsmBuilder &builder);
  void writeFunctionStart(as::AsmBuilder &builder);

  void moveRegisters(as::AsmBuilder &builder, llvm::ArrayRef<as::Register> srcs,
                     llvm::ArrayRef<as::Register> dsts);

  // return anonymous registers which means stack points
  llvm::SmallVector<as::Register> saveCallerSavedRegisters(
      as::AsmBuilder &builder,
      llvm::ArrayRef<std::pair<as::Register, as::DataSize>> datas);

  void loadCallerSavedRegisters(
      as::AsmBuilder &builder, llvm::ArrayRef<as::Register> stackpointers,
      llvm::ArrayRef<std::pair<as::Register, as::DataSize>> datas);

  LiveRangeAnalysis *getLiveRangeAnalysis() const { return liveRangeAnalysis; }
  LivenessAnalysis *getLivenessAnalysis() const { return livenessAnalysis; }
  SpillAnalysis *getSpillAnalysis() const { return spillAnalysis; }
  RegisterAllocation *getRegisterAllocation() { return &regAlloc; }
  ir::Function *getFunction() const { return function; }

  std::optional<as::Register> getSpillMemory(ir::Value value) const {
    auto liveRange = liveRangeAnalysis->getLiveRange(value);
    auto it = spillMemories.find(liveRange);
    if (it != spillMemories.end())
      return it->second;
    return std::nullopt;
  }

  std::string getAnonRegLabel() {
    return std::format("{}_anon{}", function->getName().str(), anonRegIndex++);
  }

  llvm::ArrayRef<as::Register> getIntArgRegisters() const {
    return intArgRegisters;
  }
  llvm::ArrayRef<as::Register> getFloatArgRegisters() const {
    return floatArgRegisters;
  }

  FunctionStack *getStack() { return &stack; }

  as::Register createAnonRegister(as::RegisterType regType,
                                  const StackPoint &sp);

private:
  void init();
  void saveCalleeSavedRegisters(as::AsmBuilder &builder);
  void loadCalleeSavedRegisters(as::AsmBuilder &builder);

  IRTranslater *irTranslater;
  TranslateContext *context;
  ir::Module *module;
  ir::Function *function;

  LiveRangeAnalysis *liveRangeAnalysis;
  LivenessAnalysis *livenessAnalysis;
  SpillAnalysis *spillAnalysis;
  RegisterAllocation regAlloc;
  FunctionStack stack;

  llvm::DenseMap<LiveRange, as::Register> spillMemories;
  llvm::DenseMap<LiveRange, size_t> liveRangeToIndexMap;
  llvm::DenseMap<as::Register, std::pair<StackPoint, as::DataSize>>
      anonymousRegisterToSp;

  llvm::SmallVector<as::Register, 8> intArgRegisters;
  llvm::SmallVector<as::Register, 8> floatArgRegisters;

  llvm::SmallVector<std::pair<as::Register, as::Register>> calleeSaveInfo;
  llvm::SmallVector<as::Register> localVariablesInfo;
  llvm::SmallVector<as::Register> functionIntArgMemories;
  llvm::SmallVector<as::Register> functionFloatArgMemories;
  std::optional<as::Register> returnAddressMemory;
  size_t anonRegIndex = 0;
  bool hasCall = false;
};

class TranslateRuleSet {
public:
  TranslateRuleSet();
  ~TranslateRuleSet();

  void addRule(TypeID id, std::unique_ptr<TranslationRule> rule);

  TranslationRule *getRule(TypeID id) const;

private:
  llvm::DenseMap<TypeID, std::unique_ptr<TranslationRule>> rules;
};

class TranslationRule {
public:
  TranslationRule(TypeID id) : typeId(id) {}

  virtual ~TranslationRule() = default;
  virtual utils::LogicalResult translate(as::AsmBuilder &builder,
                                         FunctionTranslater &translater,
                                         ir::InstructionStorage *inst) = 0;

  virtual bool restoreActively() const { return false; }
  virtual bool callFunction() const { return false; }

  TypeID getId() const { return typeId; }

private:
  TypeID typeId;
};

template <typename ConcreteInst>
class InstructionTranslationRule : public TranslationRule {
public:
  InstructionTranslationRule() : TranslationRule(TypeID::get<ConcreteInst>()) {}

  utils::LogicalResult translate(as::AsmBuilder &builder,
                                 FunctionTranslater &translater,
                                 ir::InstructionStorage *inst) override final {
    auto concreteInst = llvm::dyn_cast<ConcreteInst>(inst);
    if (!concreteInst)
      return utils::LogicalResult::failure();
    return translate(builder, translater, concreteInst);
  }

  virtual utils::LogicalResult translate(as::AsmBuilder &builder,
                                         FunctionTranslater &translater,
                                         ConcreteInst inst) = 0;
};

as::Immediate *getImmediate(int64_t value);
as::Immediate *getImmediate(ir::ConstantAttr constAttr);
as::Immediate *getImmediate(ir::inst::Constant constant);
as::DataSize getDataSize(ir::Type type);

as::Immediate *getImmOrLoad(as::AsmBuilder &builder, as::Register rd,
                            std::int32_t value);
void loadInt(as::AsmBuilder &builder, FunctionTranslater &translater,
             as::Register rd, std::int64_t value, ir::IntT intT);

void storeData(as::AsmBuilder &builder, FunctionTranslater &translater,
               as::Register rd, as::Register rs, as::DataSize dataSize,
               std::int64_t offset);

void loadData(as::AsmBuilder &builder, FunctionTranslater &translater,
              as::Register rd, as::Register rs, as::DataSize dataSize,
              std::int64_t offset);

template <typename Rule, typename... Args>
void registerTranslationRule(TranslateContext *context, Args &&...args) {
  TranslateRuleSet *ruleSet = context->getTranslateRuleSet();
  auto rule = std::make_unique<Rule>(std::forward<Args>(args)...);
  ruleSet->addRule(rule->getId(), std::move(rule));
}

#define KECC_UNREACHABLE_LABEL "kecc.unreachable"

void registerDefaultTranslationRules(TranslateContext *context);

} // namespace kecc

#endif // KECC_TRANSLATER_H
