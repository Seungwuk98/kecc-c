#include "kecc/asm/Register.h"
#include "kecc/translate/IRTranslater.h"
#include "kecc/translate/LiveRangeAnalyses.h"

namespace kecc {

namespace {

static void callPreparation(as::AsmBuilder &builder,
                            FunctionTranslater &translater,
                            llvm::ArrayRef<ir::Operand> operands) {
  llvm::SmallVector<as::Register, 8> operandIntRegs;
  llvm::SmallVector<as::Register, 8> operandFloatRegs;

  for (const auto &operand : operands) {
    assert(!operand.isConstant() &&
           "Constant operands should be handled separately");
    auto reg = translater.getRegister(operand);
    if (auto spillMem = translater.getSpillMemory(operand))
      reg = *spillMem;

    if (operand.getType().isa<ir::FloatT>())
      operandFloatRegs.emplace_back(reg);
    else
      operandIntRegs.emplace_back(reg);
  }

  llvm::SmallVector<as::Register, 8> intArgRegs;
  llvm::SmallVector<as::Register, 8> floatArgRegs;

  size_t argMemorySize = 0;
  FunctionStack *stack = translater.getStack();
  for (auto idx = 0u; idx < operands.size(); ++idx) {
    auto type = operands[idx].getType();
    if (type.isa<ir::FloatT>()) {
      if (floatArgRegs.size() >= translater.getFloatArgRegisters().size()) {
        auto sp = stack->callArgument(argMemorySize);
        argMemorySize +=
            translater.getModule()->getContext()->getArchitectureBitSize();
        auto anonReg =
            translater.createAnonRegister(as::RegisterType::FloatingPoint, sp);
        floatArgRegs.emplace_back(anonReg);
      } else {
        floatArgRegs.emplace_back(
            translater.getFloatArgRegisters()[floatArgRegs.size()]);
      }

    } else {
      if (intArgRegs.size() >= translater.getIntArgRegisters().size()) {
        // argument memory
        auto sp = stack->callArgument(argMemorySize);
        argMemorySize +=
            translater.getModule()->getContext()->getArchitectureBitSize();
        auto anonReg =
            translater.createAnonRegister(as::RegisterType::Integer, sp);
        intArgRegs.emplace_back(anonReg);
      } else {
        intArgRegs.emplace_back(
            translater.getIntArgRegisters()[intArgRegs.size()]);
      }
    }
  }

  stack->setCallArgumentsSize(
      std::max(argMemorySize, stack->getCallArgumentsSize()));

  translater.moveRegisters(builder, operandIntRegs, intArgRegs);
  translater.moveRegisters(builder, operandFloatRegs, floatArgRegs);
}

static void finalizeCall(as::AsmBuilder &builder,
                         FunctionTranslater &translater,
                         llvm::ArrayRef<ir::Type> retTypes,
                         llvm::ArrayRef<as::Register> resultRegs) {
  static llvm::ArrayRef<as::Register> intRetRegs = {as::Register::a0(),
                                                    as::Register::a1()};
  static llvm::ArrayRef<as::Register> floatRetRegs = {as::Register::fa0(),
                                                      as::Register::fa1()};

  llvm::SmallVector<as::Register, 8> intFromCallee;
  llvm::SmallVector<as::Register, 8> floatFromCallee;

  for (auto idx = 0u, intIdx = 0u, floatIdx = 0u; idx < retTypes.size();
       ++idx) {
    if (retTypes[idx].isa<ir::FloatT>())
      floatFromCallee.emplace_back(floatRetRegs[floatIdx++]);
    else
      intFromCallee.emplace_back(intRetRegs[intIdx++]);
  }

  llvm::SmallVector<as::Register, 8> intResultRegs;
  llvm::SmallVector<as::Register, 8> floatResultRegs;

  for (auto idx = 0u, intIdx = 0u, floatIdx = 0u; idx < resultRegs.size();
       ++idx) {
    if (resultRegs[idx].isFloatingPoint())
      floatResultRegs.emplace_back(resultRegs[idx]);
    else
      intResultRegs.emplace_back(resultRegs[idx]);
  }

  translater.moveRegisters(builder, intFromCallee, intResultRegs);
  translater.moveRegisters(builder, floatFromCallee, floatResultRegs);
}

} // namespace

class CallTranslationRule : public InstructionTranslationRule<ir::inst::Call> {
public:
  CallTranslationRule() {}

  bool restoreActively() const override { return true; }
  bool callFunction() const override {
    return true; // Caller-saved registers should be saved before call
  }

  utils::LogicalResult translate(as::AsmBuilder &builder,
                                 FunctionTranslater &translater,
                                 ir::inst::Call inst) override {
    const auto *callee = &inst.getFunctionAsOperand();

    as::Register reg = [&]() {
      if (translater.getSpillMemory(*callee))
        return translater.restoreOperand(builder, callee);
      else
        return translater.getRegister(*callee);
    }();
    callPreparation(builder, translater, inst.getArguments());

    builder.create<as::pseudo::Jalr>(reg);

    llvm::SmallVector<as::Register, 4> retRegs;
    for (ir::Value value : inst->getResults()) {
      retRegs.emplace_back(translater.getRegister(value));
    }

    finalizeCall(builder, translater,
                 callee->getType()
                     .cast<ir::PointerT>()
                     .getPointeeType()
                     .cast<ir::FunctionT>()
                     .getReturnTypes(),
                 retRegs);

    return utils::LogicalResult::success();
  }
};

class InlineCallTranslationRule
    : public InstructionTranslationRule<ir::inst::InlineCall> {
public:
  InlineCallTranslationRule() {}

  bool restoreActively() const override { return true; }
  bool callFunction() const override { return true; }

  utils::LogicalResult translate(as::AsmBuilder &builder,
                                 FunctionTranslater &translater,
                                 ir::inst::InlineCall inst) override {
    llvm::StringRef callee = inst.getName();
    ir::FunctionT funcT = inst.getFunctionType();

    callPreparation(builder, translater, inst.getArguments());
    builder.create<as::pseudo::Call>(callee);

    llvm::SmallVector<as::Register, 4> retRegs;
    for (ir::Value value : inst->getResults()) {
      retRegs.emplace_back(translater.getRegister(value));
    }

    finalizeCall(builder, translater, funcT.getReturnTypes(), retRegs);
    return utils::LogicalResult::success();
  }
};

} // namespace kecc
