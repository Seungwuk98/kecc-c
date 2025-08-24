#include "kecc/translate/IRTranslater.h"
#include "kecc/asm/AsmBuilder.h"
#include "kecc/asm/AsmInstruction.h"
#include "kecc/asm/Register.h"
#include "kecc/ir/IRInstructions.h"
#include "kecc/ir/IRTypes.h"
#include "kecc/ir/Instruction.h"
#include "kecc/ir/WalkSupport.h"
#include "kecc/translate/InterferenceGraph.h"
#include "kecc/translate/SpillAnalysis.h"
#include <format>

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

  init();
}

void FunctionTranslater::init() {
  liveRangeToIndexMap =
      liveRangeAnalysis->getCurrLRIdMap(spillAnalysis->getSpillInfo());

  // initialize `hasCall`
  function->walk([&](ir::InstructionStorage *inst) -> ir::WalkResult {
    hasCall |= inst->hasTrait<ir::CallLike>();
    return ir::WalkResult::advance();
  });

  if (hasCall) {
    stack.setReturnAddressSize(module->getContext()->getArchitectureByteSize());
    auto sp = stack.returnAddress(0);
    auto anon = createAnonRegister(as::RegisterType::Integer, sp);
    anonymousRegisterToSp.try_emplace(anon, sp);
    returnAddressMemory = anon;
  }

  llvm::SmallVector<LiveRange> spillLiveRange;
  llvm::SmallVector<as::Register> calleeSavedRegisters;
  for (const auto &[liveRange, _] : liveRangeToIndexMap) {
    auto reg = regAlloc.getRegister(function, liveRange);
    if (reg.isCalleeSaved())
      calleeSavedRegisters.emplace_back(reg);

    if (spillAnalysis->getSpillInfo().spilled.contains(liveRange))
      spillLiveRange.emplace_back(liveRange);
  }

  // initialize callee saved registers
  llvm::sort(calleeSavedRegisters, [](as::Register a, as::Register b) {
    auto aFloat = a.isFloatingPoint();
    auto bFloat = b.isFloatingPoint();
    if (aFloat != bFloat)
      return aFloat < bFloat;
    return a.getABIIndex() < b.getABIIndex();
  });

  size_t totalCalleeSavedSize = 0;
  for (const auto &reg : calleeSavedRegisters) {
    auto sp = stack.calleeSavedRegister(totalCalleeSavedSize);
    totalCalleeSavedSize += module->getContext()->getArchitectureByteSize();

    auto anon = createAnonRegister(as::RegisterType::Integer, sp);
    anonymousRegisterToSp.try_emplace(anon, sp);
  }

  stack.setCalleeSavedRegistersSize(totalCalleeSavedSize);

  // initialize local variables
  size_t totalLocalVarSize = 0;
  for (ir::InstructionStorage *storage : *function->getAllocationBlock()) {
    ir::inst::LocalVariable localVar =
        storage->getDefiningInst<ir::inst::LocalVariable>();
    assert(localVar &&
           "LocalVariable instruction must be defined in the allocation block");

    auto type = localVar.getType().cast<ir::PointerT>().getPointeeType();
    auto dataSize = getDataSize(type);
    auto sp = stack.localVariable(totalLocalVarSize);
    totalLocalVarSize += dataSize.getByteSize();
    auto anon = createAnonRegister(as::RegisterType::Integer, sp);
    anonymousRegisterToSp.try_emplace(anon, std::make_pair(sp, dataSize));
    localVariablesInfo.emplace_back(anon);
  }
  stack.setLocalVariablesSize(totalLocalVarSize);

  // initialize spill

  size_t argIntIndex = 0u;
  size_t argFloatIndex = 0u;
  size_t offset = 0u;
  for (ir::InstructionStorage *inst : *function->getEntryBlock()) {
    ir::inst::FunctionArgument arg =
        inst->getDefiningInst<ir::inst::FunctionArgument>();
    if (!arg)
      break;

    auto liveRange = liveRangeAnalysis->getLiveRange(function, arg);

    if (arg.getType().isa<ir::FloatT>()) {
      if (argFloatIndex >= getFloatArgRegisters().size()) {
        auto sp = stack.functionArgument(offset);
        offset += module->getContext()->getArchitectureByteSize();
        auto anon = createAnonRegister(as::RegisterType::Integer, sp);
        anonymousRegisterToSp.try_emplace(anon, sp);

        if (isSpilled(liveRange)) {
          spillMemories.try_emplace(liveRange, anon);
        }

        functionFloatArgMemories.emplace_back(anon);
      }
      argFloatIndex++;
    } else {
      if (argIntIndex >= getIntArgRegisters().size()) {
        auto sp = stack.functionArgument(offset);
        offset += module->getContext()->getArchitectureByteSize();
        auto anon = createAnonRegister(as::RegisterType::Integer, sp);
        anonymousRegisterToSp.try_emplace(anon, sp);

        if (isSpilled(liveRange)) {
          spillMemories.try_emplace(liveRange, anon);
        }

        functionIntArgMemories.emplace_back(anon);
      }
      argIntIndex++;
    }
  }

  llvm::sort(spillLiveRange, [&](const LiveRange &a, const LiveRange &b) {
    return liveRangeToIndexMap[a] < liveRangeToIndexMap[b];
  });

  size_t totalSpillSize = 0;
  for (const auto &liveRange : spillLiveRange) {
    if (spillMemories.contains(liveRange))
      continue; // Already processed

    auto type = liveRangeAnalysis->getLiveRangeType(function, liveRange);
    auto dataSize = getDataSize(type);
    auto sp = stack.spilledRegister(totalSpillSize);
    totalSpillSize += dataSize.getByteSize();
    auto anon = createAnonRegister(as::RegisterType::Integer, sp);
    anonymousRegisterToSp.try_emplace(anon, std::make_pair(sp, dataSize));
    spillMemories.try_emplace(liveRange, anon);
  }

  stack.setSpilledRegistersSize(totalCalleeSavedSize);
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
    auto anon = createAnonRegister(as::RegisterType::Integer, sp);
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
  builder.setInsertionPointStart(funcEntryBlock);
  writeFunctionStart(builder);

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

void FunctionTranslater::writeFunctionStart(as::AsmBuilder &builder) {
  auto tempReg = getTranslateContext()->getTempRegisters()[0];

  if (stack.getTotalSize() > 0) {
    auto *imm = getImmOrLoad(builder, tempReg,
                             -static_cast<std::int32_t>(stack.getTotalSize()));
    if (imm) {
      builder.create<as::itype::Addi>(as::Register::sp(), as::Register::sp(),
                                      *imm, as::DataSize::doubleWord());
    } else {
      builder.create<as::rtype::Add>(as::Register::sp(), as::Register::sp(),
                                     tempReg, as::DataSize::doubleWord());
    }
  }

  if (stack.getReturnAddressSize() > 0) {
    auto sp = stack.returnAddress(0);
    auto anon = createAnonRegister(as::RegisterType::Integer, sp);
    anonymousRegisterToSp.try_emplace(anon, sp);
    builder.create<as::stype::Store>(as::Register::ra(), anon, getImmediate(0),
                                     as::DataSize::doubleWord());
  }

  saveCalleeSavedRegisters(builder);

  auto localVarIdx = 0u;
  for (ir::InstructionStorage *inst : *function->getAllocationBlock()) {
    auto localVar = inst->getDefiningInst<ir::inst::LocalVariable>();
    assert(localVar && "LocalVariable instruction must be defined in the "
                       "allocation block");
    auto liveRange = liveRangeAnalysis->getLiveRange(function, localVar);

    auto rd = getRegister(localVar);
    builder.create<as::pseudo::Mv>(rd, localVariablesInfo[localVarIdx++]);
    if (isSpilled(liveRange)) {
      spillRegister(builder, liveRange);
    }
  }

  // handle arguments
  size_t argIntIndex = 0u;
  size_t argFloatIndex = 0u;
  for (ir::InstructionStorage *inst : *function->getEntryBlock()) {
    auto arg = inst->getDefiningInst<ir::inst::FunctionArgument>();
    if (!arg)
      break;

    auto liveRange = liveRangeAnalysis->getLiveRange(function, arg);
    auto rd = getRegister(arg);
    auto dataSize = getDataSize(arg.getType());

    if (arg.getType().isa<ir::FloatT>()) {
      if (argFloatIndex >= getFloatArgRegisters().size()) {
        auto sp = functionFloatArgMemories[argFloatIndex -
                                           getFloatArgRegisters().size()];
        builder.create<as::itype::Load>(rd, sp, getImmediate(0), dataSize);
        // If the argument is spilled, we don't need to spill it again because
        // it already has a memory from caller
      } else {
        auto argReg = getFloatArgRegisters()[argFloatIndex];
        assert(argReg == rd &&
               "Argument register must match the register allocated for the "
               "argument");
        if (isSpilled(liveRange)) {
          spillRegister(builder, liveRange);
        }
      }
      argFloatIndex++;
    } else {
      if (argIntIndex >= getIntArgRegisters().size()) {
        auto sp =
            functionIntArgMemories[argIntIndex - getIntArgRegisters().size()];
        builder.create<as::itype::Load>(rd, sp, getImmediate(0), dataSize);
        // If the argument is spilled, we don't need to spill it again because
        // it already has a memory from caller
      } else {
        auto argReg = getIntArgRegisters()[argIntIndex];
        assert(argReg == rd &&
               "Argument register must match the register allocated for the "
               "argument");
        if (isSpilled(liveRange)) {
          spillRegister(builder, liveRange);
        }
      }
      argIntIndex++;
    }
  }
}

void FunctionTranslater::writeFunctionEnd(as::AsmBuilder &builder) {
  // Restore callee saved registers
  loadCalleeSavedRegisters(builder);

  // Restore return address
  if (stack.getReturnAddressSize() > 0) {
    auto sp = *returnAddressMemory;
    builder.create<as::itype::Load>(as::Register::ra(), sp, getImmediate(0),
                                    as::DataSize::doubleWord());
  }

  // stack cleanup
  if (stack.getTotalSize() > 0) {
    auto tempReg = getTranslateContext()->getTempRegisters()[0];
    auto *imm = getImmOrLoad(builder, tempReg, stack.getTotalSize());
    if (imm) {
      builder.create<as::itype::Addi>(as::Register::sp(), as::Register::sp(),
                                      *imm, as::DataSize::doubleWord());
    } else {
      builder.create<as::rtype::Add>(as::Register::sp(), as::Register::sp(),
                                     tempReg, as::DataSize::doubleWord());
    }
  }
}

void FunctionTranslater::spillRegister(as::AsmBuilder &builder,
                                       LiveRange liveRange) {
  auto memory = spillMemories.at(liveRange);
  auto type = liveRangeAnalysis->getLiveRangeType(function, liveRange);
  auto dataSize = getDataSize(type);
  auto reg = regAlloc.getRegister(function, liveRange);

  builder.create<as::stype::Store>(memory, reg, getImmediate(0), dataSize);
}

as::Register FunctionTranslater::restoreOperand(as::AsmBuilder &builder,
                                                const ir::Operand *operand) {
  auto restoredLiveRange = spillAnalysis->getSpillInfo().restore.at(operand);
  auto spilledLiveRange = liveRangeAnalysis->getLiveRange(function, *operand);
  assert(!isSpilled(restoredLiveRange) && "Restored value must not be spilled");

  auto spilledMemory = spillMemories.at(spilledLiveRange);
  auto type = operand->getType();
  auto dataSize = getDataSize(type);
  auto rd = regAlloc.getRegister(function, restoredLiveRange);

  builder.create<as::itype::Load>(rd, spilledMemory, getImmediate(0), dataSize);
  return rd;
}

void FunctionTranslater::saveCalleeSavedRegisters(as::AsmBuilder &builder) {
  for (const auto &[reg, sp] : calleeSaveInfo) {
    auto dataSize = reg.isFloatingPoint() ? as::DataSize::doublePrecision()
                                          : as::DataSize::doubleWord();
    builder.create<as::stype::Store>(sp, reg, getImmediate(0), dataSize);
  }
}

void FunctionTranslater::loadCalleeSavedRegisters(as::AsmBuilder &builder) {
  for (const auto &[reg, sp] : calleeSaveInfo) {
    auto dataSize = reg.isFloatingPoint() ? as::DataSize::doublePrecision()
                                          : as::DataSize::doubleWord();
    builder.create<as::itype::Load>(reg, sp, getImmediate(0), dataSize);
  }
}

} // namespace kecc
