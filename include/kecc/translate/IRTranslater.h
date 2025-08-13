#ifndef KECC_TRANSLATER_H
#define KECC_TRANSLATER_H

#include "kecc/asm/Asm.h"
#include "kecc/asm/AsmBuilder.h"
#include "kecc/asm/AsmInstruction.h"
#include "kecc/ir/IRInstructions.h"
#include "kecc/ir/Module.h"
#include "kecc/translate/InterferenceGraph.h"
#include "kecc/translate/RegisterAllocation.h"
#include "kecc/translate/TranslateContext.h"
#include "kecc/utils/LogicalResult.h"

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
  FunctionTranslater(TranslateContext *context, ir::Module *module,
                     ir::Function *function);

  as::Block *createBlock(ir::Block *block);

  as::Function *translate();

  TranslateContext *getContext() const { return context; }

  RegisterAllocation &regAlloc();

private:
  TranslateContext *context;
  ir::Module *module;
  ir::Function *function;
};

class TranslateRuleSet {
public:
  TranslateRuleSet();
  ~TranslateRuleSet();

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

  TypeID getId() const { return typeId; }
  int getBenefit() const { return benefit; }

private:
  TypeID typeId;
  int benefit;
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

as::Immediate *getImmediate(ir::inst::Constant constant);
as::DataSize getDataSize(ir::Type type);

} // namespace kecc

#endif // KECC_TRANSLATER_H
