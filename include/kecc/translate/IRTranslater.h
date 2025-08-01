#ifndef KECC_TRANSLATER_H
#define KECC_TRANSLATER_H

#include "kecc/asm/Asm.h"
#include "kecc/asm/AsmBuilder.h"
#include "kecc/ir/Module.h"
#include "kecc/translate/TranslateContext.h"
#include "kecc/utils/LogicalResult.h"

namespace kecc {

class TranslateRule;

class IRTranslater {
public:
  IRTranslater(TranslateContext *context, ir::Module *module)
      : context(context), module(module) {}

  as::Asm *translate();

  TranslateContext *getContext() const { return context; }
  ir::Module *getModule() const { return module; }

private:
  TranslateContext *context;
  ir::Module *module;
};

class FuncTranslater {
public:
private:
};

class TranslateRuleSet {
public:
  TranslateRuleSet();
  ~TranslateRuleSet();

private:
  llvm::DenseMap<TypeID, std::unique_ptr<TranslateRule>> rules;
};

class TranslateRule {
public:
  TranslateRule(TypeID id) : typeId(id) {}

  virtual ~TranslateRule() = default;
  virtual utils::LogicalResult translate(as::AsmBuilder &builder,
                                         FuncTranslater &translater,
                                         as::Instruction *inst) = 0;

  TypeID getId() const { return typeId; }
  int getBenefit() const { return benefit; }

private:
  TypeID typeId;
  int benefit;
};

template <typename ConcreteInst>
class InstructionTranslateRule : public TranslateRule {
public:
  InstructionTranslateRule() : TranslateRule(TypeID::get<ConcreteInst>()) {}

  utils::LogicalResult translate(as::AsmBuilder &builder,
                                 FuncTranslater &translater,
                                 as::Instruction *inst) override final {
    auto *concreteInst = inst->dyn_cast<ConcreteInst>();
    if (!concreteInst)
      return utils::LogicalResult::failure();
    return translate(builder, translater, concreteInst);
  }

  virtual utils::LogicalResult translate(as::AsmBuilder &builder,
                                         FuncTranslater &translater,
                                         ConcreteInst *inst) = 0;
};

} // namespace kecc

#endif // KECC_TRANSLATER_H
