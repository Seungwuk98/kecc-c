#ifndef KECC_TRANSLATER_H
#define KECC_TRANSLATER_H

#include "kecc/asm/Asm.h"
#include "kecc/asm/AsmBuilder.h"
#include "kecc/asm/AsmInstruction.h"
#include "kecc/ir/IRAttributes.h"
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
struct DataSpace {
  enum Kind { Value, Memory };

  static DataSpace value(as::Register reg) { return DataSpace(Value, reg); }
  static DataSpace memory(as::Register reg) {
    assert(reg.isAnonymous() && "Spill memory must be anonymous register");
    return DataSpace(Memory, reg);
  }

  bool isValue() const { return kind == Value; }
  bool isMemory() const { return kind == Memory; }
  as::Register getRegister() const { return reg; }

  friend llvm::hash_code hash_value(const DataSpace &data) {
    return llvm::hash_combine(
        static_cast<unsigned>(data.isValue()),
        llvm::DenseMapInfo<as::Register>::getHashValue(data.reg));
  }

private:
  friend struct llvm::DenseMapInfo<DataSpace>;
  DataSpace(Kind kind, as::Register data) : kind(kind), reg(std::move(data)) {}
  Kind kind;
  as::Register reg;
};
} // namespace kecc

namespace llvm {

template <> struct DenseMapInfo<kecc::DataSpace> {
  static inline kecc::DataSpace getEmptyKey() {
    return kecc::DataSpace(static_cast<kecc::DataSpace::Kind>(-1),
                           DenseMapInfo<kecc::as::Register>::getEmptyKey());
  }

  static inline kecc::DataSpace getTombstoneKey() {
    return kecc::DataSpace(static_cast<kecc::DataSpace::Kind>(-1),
                           DenseMapInfo<kecc::as::Register>::getTombstoneKey());
  }

  static unsigned getHashValue(const kecc::DataSpace &data) {
    return hash_combine(
        static_cast<unsigned>(data.isValue()),
        DenseMapInfo<kecc::as::Register>::getHashValue(data.getRegister()));
  }

  static bool isEqual(const kecc::DataSpace &lhs, const kecc::DataSpace &rhs) {
    return lhs.isValue() == rhs.isValue() &&
           lhs.getRegister() == rhs.getRegister();
  }
};
} // namespace llvm

namespace kecc {

class TranslationRule;

void defaultRegisterSetup(TranslateContext *context);

class IRTranslater {
public:
  IRTranslater(TranslateContext *context, ir::Module *module)
      : context(context), module(module), regAlloc(module, context) {
    init();
  }
  ~IRTranslater();

  std::unique_ptr<as::Asm> translate();

  TranslateContext *getContext() const { return context; }
  ir::Module *getModule() const { return module; }

  std::pair<llvm::StringRef, std::optional<as::DataSize>>
  getOrCreateConstantLabel(ir::ConstantAttr constant);
  std::pair<llvm::StringRef, std::optional<as::DataSize>>
  getOrCreateConstantLabel(std::int64_t value);

  RegisterAllocation &getRegisterAllocation() { return regAlloc; }

private:
  void init();

  as::Function *translateFunction(ir::Function *function);
  as::Variable *
  translateGlobalVariable(ir::inst::GlobalVariableDefinition globalVar);

  void translateGlobalVariableImpl(
      ir::Type type, ir::Attribute init, ir::InitializerAttr astAttr,
      ir::StructSizeAnalysis *structSizeAnalysis,
      llvm::SmallVectorImpl<as::Directive *> &directives, size_t &currSize);

  as::Variable *translateConstant(ir::inst::Constant constant);

  TranslateContext *context;
  ir::Module *module;
  RegisterAllocation regAlloc;
  llvm::DenseMap<llvm::StringRef, ir::Type> globalVarMap;
  llvm::DenseMap<ir::ConstantAttr,
                 std::pair<llvm::StringRef, std::optional<as::DataSize>>>
      constantToLabelMap;
  llvm::SmallVector<as::Variable *, 8> constants;
  size_t constantIndex = 0;
};

class FunctionTranslater {
private:
public:
  FunctionTranslater(IRTranslater *translater, TranslateContext *context,
                     ir::Module *module, ir::Function *function);

  static std::string getBlockName(ir::Block *block);
  as::Block *createBlock(ir::Block *block);

  as::Function *translate();

  TranslateContext *getTranslateContext() const { return context; }
  IRTranslater *getIRTranslater() const { return irTranslater; }

  as::Register getRegister(LiveRange liveRange);
  as::Register getRegister(ir::Value value);
  as::Register getOperandRegister(const ir::Operand *operand);

  std::pair<llvm::StringRef, std::optional<as::DataSize>>
  getConstantLabel(ir::ConstantAttr constant) {
    return irTranslater->getOrCreateConstantLabel(constant);
  }
  std::pair<llvm::StringRef, std::optional<as::DataSize>>
  getConstantLabel(std::int64_t value) {
    return irTranslater->getOrCreateConstantLabel(value);
  }

  ir::Module *getModule() const { return module; }

  as::Register restoreOperand(as::AsmBuilder &builder,
                              const ir::Operand *operand);
  void spillRegister(as::AsmBuilder &builder, LiveRange liveRange);
  bool isSpilled(LiveRange liveRange) const;

  void writeFunctionEnd(as::AsmBuilder &builder);
  void writeFunctionEndImpl(as::AsmBuilder &builder);
  std::string functionEndLabel() const;
  void writeFunctionStart(as::AsmBuilder &builder);

  // move srcs to dsts
  // If there are anonymout registers, it handles them as 8byte memory
  void moveRegisters(as::AsmBuilder &builder, llvm::ArrayRef<DataSpace> srcs,
                     llvm::ArrayRef<DataSpace> dsts);

  // return anonymous registers which means stack points
  llvm::SmallVector<as::Register> saveCallerSavedRegisters(
      as::AsmBuilder &builder,
      llvm::ArrayRef<std::pair<as::Register, ir::Type>> datas);

  void loadCallerSavedRegisters(
      as::AsmBuilder &builder, llvm::ArrayRef<as::Register> stackpointers,
      llvm::ArrayRef<std::pair<as::Register, ir::Type>> datas);

  LiveRangeAnalysis *getLiveRangeAnalysis() const { return liveRangeAnalysis; }
  LivenessAnalysis *getLivenessAnalysis() const { return livenessAnalysis; }
  SpillAnalysis *getSpillAnalysis() const { return spillAnalysis; }
  ir::Function *getFunction() const { return function; }

  std::optional<DataSpace> getSpillData(ir::Value value) const {
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
                                  const StackPoint &sp,
                                  std::optional<as::DataSize> dataSize,
                                  bool isSigned);

  bool hasMultipleReturn() const { return multipleReturn; }

private:
  void init();
  void saveCalleeSavedRegisters(as::AsmBuilder &builder);
  void loadCalleeSavedRegisters(as::AsmBuilder &builder);

  void substitueAnonymousRegisters(as::Function *func);

  IRTranslater *irTranslater;
  TranslateContext *context;
  ir::Module *module;
  ir::Function *function;

  LiveRangeAnalysis *liveRangeAnalysis;
  LivenessAnalysis *livenessAnalysis;
  SpillAnalysis *spillAnalysis;
  FunctionStack stack;

  llvm::DenseMap<LiveRange, DataSpace> spillMemories;
  llvm::DenseMap<LiveRange, size_t> liveRangeToIndexMap;
  llvm::DenseMap<as::Register,
                 std::tuple<StackPoint, std::optional<as::DataSize>, bool>>
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
  bool multipleReturn = false;
};

class TranslationRuleSet {
public:
  TranslationRuleSet();
  ~TranslationRuleSet();

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
  virtual bool spillActively() const { return false; }
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
void loadInt32(as::AsmBuilder &builder, as::Register rd, std::int32_t value);
void loadInt64(as::AsmBuilder &builder, FunctionTranslater &translater,
               as::Register rd, std::int64_t value);

void loadInt(as::AsmBuilder &builder, FunctionTranslater &translater,
             as::Register rd, std::int64_t value, ir::IntT intT);

void loadFloat(as::AsmBuilder &builder, FunctionTranslater &translater,
               as::Register rd, llvm::APFloat value);

void storeData(as::AsmBuilder &builder, FunctionTranslater &translater,
               as::Register rd, as::Register rs, as::DataSize dataSize,
               std::int32_t offset);

void loadData(as::AsmBuilder &builder, FunctionTranslater &translater,
              as::Register rd, as::Register rs, as::DataSize dataSize,
              std::int32_t offset, bool isSigned);

template <typename Rule, typename... Args>
void registerTranslationRule(TranslateContext *context, Args &&...args) {
  TranslationRuleSet *ruleSet = context->getTranslateRuleSet();
  auto rule = std::make_unique<Rule>(std::forward<Args>(args)...);
  ruleSet->addRule(rule->getId(), std::move(rule));
}

#define KECC_UNREACHABLE_LABEL "kecc.unreachable"

void registerDefaultTranslationRules(TranslateContext *context);

} // namespace kecc

#endif // KECC_TRANSLATER_H
