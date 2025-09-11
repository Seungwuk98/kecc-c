#include "kecc/translate/IRTranslater.h"
#include "kecc/asm/Asm.h"
#include "kecc/asm/AsmBuilder.h"
#include "kecc/asm/AsmInstruction.h"
#include "kecc/asm/Register.h"
#include "kecc/ir/IRAnalyses.h"
#include "kecc/ir/IRAttributes.h"
#include "kecc/ir/IRInstructions.h"
#include "kecc/ir/IRTypes.h"
#include "kecc/ir/Instruction.h"
#include "kecc/ir/TypeAttributeSupport.h"
#include "kecc/ir/WalkSupport.h"
#include "kecc/translate/InterferenceGraph.h"
#include "kecc/translate/MoveSchedule.h"
#include "kecc/translate/SpillAnalysis.h"
#include "kecc/utils/Diag.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/SourceMgr.h"
#include <format>

namespace kecc {

void defaultRegisterSetup(TranslateContext *context) {
  static llvm::SmallVector<as::Register> tempIntRegs = {
      as::Register::t0(),
      as::Register::t1(),
  };

  context->setTempRegisters(tempIntRegs);

  static llvm::SmallVector<as::Register> intRegs = {
      as::Register::a0(), as::Register::a1(),  as::Register::a2(),
      as::Register::a3(), as::Register::a4(),  as::Register::a5(),
      as::Register::a6(), as::Register::a7(),  as::Register::t2(),

      as::Register::t3(), as::Register::t4(),  as::Register::t5(),
      as::Register::t6(),

      as::Register::s0(), as::Register::s1(),  as::Register::s2(),
      as::Register::s3(), as::Register::s4(),  as::Register::s5(),
      as::Register::s6(), as::Register::s7(),  as::Register::s8(),
      as::Register::s9(), as::Register::s10(), as::Register::s11(),
  };

  static llvm::SmallVector<as::Register> floatRegs = {
      as::Register::fa0(), as::Register::fa1(),  as::Register::fa2(),
      as::Register::fa3(), as::Register::fa4(),  as::Register::fa5(),
      as::Register::fa6(), as::Register::fa7(),

      as::Register::ft0(), as::Register::ft1(),  as::Register::ft2(),
      as::Register::ft3(), as::Register::ft4(),  as::Register::ft5(),
      as::Register::ft6(), as::Register::ft7(),  as::Register::ft8(),
      as::Register::ft9(), as::Register::ft10(), as::Register::ft11(),

      as::Register::fs0(), as::Register::fs1(),  as::Register::fs2(),
      as::Register::fs3(), as::Register::fs4(),  as::Register::fs5(),
      as::Register::fs6(), as::Register::fs7(),  as::Register::fs8(),
      as::Register::fs9(), as::Register::fs10(), as::Register::fs11(),
  };

  context->setRegistersForAllocate(intRegs, floatRegs);
}

IRTranslater::~IRTranslater() {
  for (as::Variable *constant : constants) {
    delete constant;
  }
}

void IRTranslater::init() {
  for (ir::InstructionStorage *inst : *module->getIR()->getGlobalBlock()) {
    auto globalVar =
        inst->getDefiningInst<ir::inst::GlobalVariableDefinition>();
    assert(globalVar && "Expected global variable");
    auto [_, inserted] =
        globalVarMap.try_emplace(globalVar.getName(), globalVar.getType());
    assert(inserted && "Duplicate global variable name");
    (void)inserted;
  }
}

std::pair<llvm::StringRef, std::optional<as::DataSize>>
IRTranslater::getOrCreateConstantLabel(ir::ConstantAttr constant) {
  if (auto it = constantToLabelMap.find(constant);
      it != constantToLabelMap.end()) {
    return it->second;
  }

  auto label = std::format(".LC{}", constantIndex++);
  as::Variable *newVariable;
  as::DataSize dataSize = as::DataSize::doubleWord();
  if (auto intConst = constant.dyn_cast<ir::ConstantIntAttr>()) {
    auto quad = new as::QuadDirective(intConst.getValue());
    newVariable = new as::Variable(label, {quad});
  } else if (auto floatConst = constant.dyn_cast<ir::ConstantFloatAttr>()) {
    auto value = floatConst.getValue();
    auto bits = value.bitcastToAPInt();

    std::uint64_t data;
    as::Directive *directive;
    if (floatConst.getFloatType().getBitWidth() == 32) {
      data = bits.getZExtValue() & 0xFFFFFFFF;
      dataSize = as::DataSize::singlePrecision();
      directive = new as::WordDirective(static_cast<std::uint32_t>(data));
    } else if (floatConst.getFloatType().getBitWidth() == 64) {
      data = bits.getZExtValue();
      dataSize = as::DataSize::doublePrecision();
      directive = new as::QuadDirective(data);
    } else {
      llvm_unreachable("Unsupported float bit width");
    }

    newVariable = new as::Variable(label, {directive});
  } else if (auto varConst = constant.dyn_cast<ir::ConstantVariableAttr>()) {
    auto globalVarIt = globalVarMap.find(varConst.getName());
    if (globalVarIt == globalVarMap.end()) {
      // it must be function
      assert(constant.getType().isa<ir::PointerT>());
      auto ptrType = constant.getType().cast<ir::PointerT>();
      assert(ptrType.getPointeeType().isa<ir::FunctionT>());

      return {varConst.getName(), std::nullopt};
    }
    auto type = globalVarIt->getSecond();

    std::optional<as::DataSize> dataSize;
    if (!type.isa<ir::NameStruct, ir::ArrayT>())
      dataSize = getDataSize(type);

    return {varConst.getName(), dataSize};
  } else {
    llvm_unreachable("Unsupported constant type");
  }

  constants.emplace_back(newVariable);
  constantToLabelMap.try_emplace(constant, newVariable->getLabel(), dataSize);
  return {newVariable->getLabel(), dataSize};
}

std::pair<llvm::StringRef, std::optional<as::DataSize>>
IRTranslater::getOrCreateConstantLabel(int64_t value) {
  auto constInt =
      ir::ConstantIntAttr::get(module->getContext(), value, 64, true);
  return getOrCreateConstantLabel(constInt);
}

std::unique_ptr<as::Asm> IRTranslater::translate() {
  llvm::SmallVector<as::Section<as::Function> *> functions;
  llvm::SmallVector<as::Section<as::Variable> *> dataVariables;
  llvm::SmallVector<as::Section<as::Variable> *> bssVariables;

  bool fail = false;
  for (ir::InstructionStorage *inst : *module->getIR()->getGlobalBlock()) {
    auto globalVar =
        inst->getDefiningInst<ir::inst::GlobalVariableDefinition>();
    assert(globalVar && "Expected global variable");
    auto *variable = translateGlobalVariable(globalVar);
    if (!variable) {
      fail = true;
      break;
    }
    bool isData = globalVar.hasInitializer();
    (isData ? dataVariables : bssVariables)
        .emplace_back(new as::Section<as::Variable>(
            {
                new as::TypeDirective(globalVar.getName(),
                                      as::TypeDirective::Kind::Object),
                new as::SectionDirective(
                    isData ? as::SectionDirective::SectionType::Data
                           : as::SectionDirective::SectionType::Bss),
                new as::GloblDirective(globalVar.getName()),
                new as::AlignDirective(2),
            },
            variable));
  }

  if (!fail) {
    for (ir::Function *function : *module->getIR()) {
      if (!function->hasDefinition())
        continue;
      auto *asFunction = translateFunction(function);
      if (!asFunction) {
        fail = true;
        break;
      }
      assert(asFunction && "Function translation failed");

      auto *section = new as::Section<as::Function>(
          {new as::GloblDirective(function->getName()),
           new as::AlignDirective(1),
           new as::TypeDirective(function->getName(),
                                 as::TypeDirective::Kind::Function)},
          asFunction);
      functions.emplace_back(section);
    }
  }

  if (fail) {
    for (auto *func : functions)
      delete func;
    for (auto *var : dataVariables)
      delete var;
    for (auto *var : bssVariables)
      delete var;
    return nullptr;
  }

  dataVariables.append(bssVariables);
  for (as::Variable *constant : constants) {
    dataVariables.emplace_back(new as::Section<as::Variable>(
        {
            new as::TypeDirective(constant->getLabel(),
                                  as::TypeDirective::Kind::Object),
            new as::SectionDirective(as::SectionDirective::SectionType::Rodata),
            new as::AlignDirective(2),
        },
        constant));
  }
  constants.clear();

  return std::make_unique<as::Asm>(functions, dataVariables);
}

as::Function *IRTranslater::translateFunction(ir::Function *function) {
  FunctionTranslater translater(this, context, module, function);
  return translater.translate();
}

as::Variable *IRTranslater::translateGlobalVariable(
    ir::inst::GlobalVariableDefinition globalVar) {
  auto type = globalVar.getType();

  ir::StructSizeAnalysis *structAnalysis =
      module->getOrCreateAnalysis<ir::StructSizeAnalysis>(module);
  llvm::SmallVector<as::Directive *, 4> directives;
  size_t currSize = 0;
  if (auto astAttr = globalVar.getInitializer().cast<ir::InitializerAttr>()) {
    globalVar.interpretInitializer();
    if (module->getContext()->diag().hasError())
      return nullptr;

    translateGlobalVariableImpl(type, globalVar.getInitializer(), astAttr,
                                structAnalysis, directives, currSize);
  } else {
    translateGlobalVariableImpl(type, nullptr, nullptr, structAnalysis,
                                directives, currSize);
  }

  auto [size, align] = type.getSizeAndAlign(structAnalysis->getStructSizeMap());
  assert(size == currSize && "Size mismatch");
  (void)size;

  return new as::Variable(globalVar.getName(), directives);
}

void IRTranslater::translateGlobalVariableImpl(
    ir::Type type, ir::Attribute init, ir::InitializerAttr astAttr,
    ir::StructSizeAnalysis *structSizeAnalysis,
    llvm::SmallVectorImpl<as::Directive *> &directives, size_t &currSize) {

  auto isZero = !init;
  if (auto array = init.dyn_cast_or_null<ir::ArrayAttr>()) {
    if (array.getValues().empty())
      isZero = true;
  }

  if (isZero) {
    auto [size, align] =
        type.getSizeAndAlign(structSizeAnalysis->getStructSizeMap());
    if (size < align)
      size = align;

    directives.emplace_back(new as::ZeroDirective(size));
    currSize += size;
    return;
  }

  utils::DiagEngine &diag = type.getContext()->diag();
  llvm::TypeSwitch<ir::Type, void>(type)
      .Case([&](ir::IntT intT) {
        auto intAttr = init.dyn_cast<ir::ConstantIntAttr>();
        if (!intAttr) {
          diag.report(astAttr.getRange(), llvm::SourceMgr::DK_Error,
                      "Initializer type mismatch: expected integer");
          return;
        }
        auto bitWidth = intT.getBitWidth();
        auto value = intAttr.getValue();
        as::Directive *intDirective;

        as::DataSize dataSize = getDataSize(intT);
        switch (dataSize.getKind()) {
        case as::DataSize::Byte: {
          intDirective = new as::ByteDirective(value & 0xFF);
          break;
        }
        case as::DataSize::Half: {
          intDirective = new as::HalfDirective(value & 0xFFFF);
          break;
        }
        case as::DataSize::Word: {
          intDirective = new as::WordDirective(value & 0xFFFFFFFF);
          break;
        }
        case as::DataSize::Double: {
          intDirective = new as::QuadDirective(value);
          break;
        }
        default:
          llvm_unreachable("Unsupported integer data size");
        }

        directives.emplace_back(intDirective);
        currSize += dataSize.getByteSize();
      })

      .Case([&](ir::FloatT floatT) {
        auto floatAttr = init.dyn_cast<ir::ConstantFloatAttr>();
        if (!floatAttr) {
          diag.report(astAttr.getRange(), llvm::SourceMgr::DK_Error,
                      "Initializer type mismatch: expected float");
          return;
        }

        auto value = floatAttr.getValue();
        auto bits = value.bitcastToAPInt();

        std::uint64_t data;
        as::Directive *directive;
        if (floatT.getBitWidth() == 32) {
          data = bits.getZExtValue() & 0xFFFFFFFF;
          directive = new as::WordDirective(static_cast<std::uint32_t>(data));
        } else if (floatT.getBitWidth() == 64) {
          data = bits.getZExtValue();
          directive = new as::QuadDirective(data);
        } else {
          diag.report(astAttr.getRange(), llvm::SourceMgr::DK_Error,
                      "Unsupported float bit width");
          return;
        }

        directives.emplace_back(directive);
        currSize += floatT.getBitWidth() / ir::BITS_OF_BYTE;
      })

      .Case([&](ir::ArrayT arrayT) {
        auto arrayAttr = init.dyn_cast<ir::ArrayAttr>();
        if (!arrayAttr) {
          diag.report(astAttr.getRange(), llvm::SourceMgr::DK_Error,
                      "Expected array initalizer");
          return;
        }

        auto arrayInitializer = astAttr.cast<ir::ASTInitializerList>();
        auto elementT = arrayT.getElementType();
        for (const auto &[value, ast] :
             llvm::zip(arrayAttr.getValues(), arrayInitializer.getValues())) {
          translateGlobalVariableImpl(elementT, value, ast, structSizeAnalysis,
                                      directives, currSize);
        }

        if (arrayT.getSize() < arrayAttr.getValues().size()) {
          auto [size, align] =
              elementT.getSizeAndAlign(structSizeAnalysis->getStructSizeMap());
          if (size < align)
            size = align;

          auto zeroSize =
              size * (arrayT.getSize() - arrayAttr.getValues().size());
          directives.emplace_back(new as::ZeroDirective(zeroSize));
          currSize += zeroSize;
        }
      })

      .Case([&](ir::NameStruct structT) {
        auto arrayAttr = init.dyn_cast<ir::ArrayAttr>();
        if (!arrayAttr) {
          diag.report(astAttr.getRange(), llvm::SourceMgr::DK_Error,
                      "Expected array initalizer");
          return;
        }

        auto arrayInitializer = astAttr.cast<ir::ASTInitializerList>();
        const auto &[size, align, offsets] =
            structSizeAnalysis->getStructSizeMap().at(structT.getName());

        auto fields =
            structSizeAnalysis->getStructFieldsMap().at(structT.getName());

        size_t structSize = 0;
        for (const auto &[idx, field, offset] :
             llvm::enumerate(fields, offsets)) {
          if (offset != structSize) {
            auto zeroSize = offset - structSize;
            directives.emplace_back(new as::ZeroDirective(zeroSize));
            structSize += zeroSize;
          }

          translateGlobalVariableImpl(
              field,
              idx < arrayAttr.getValues().size() ? arrayAttr.getValues()[idx]
                                                 : nullptr,
              idx < arrayInitializer.getValues().size()
                  ? arrayInitializer.getValues()[idx]
                  : nullptr,
              structSizeAnalysis, directives, structSize);
        }

        currSize += structSize;
      })
      .Default(
          [&](ir::Type) { llvm_unreachable("Can't initialize this type"); });
}

FunctionTranslater::FunctionTranslater(IRTranslater *translater,
                                       TranslateContext *context,
                                       ir::Module *module,
                                       ir::Function *function)
    : irTranslater(translater), context(context), module(module),
      function(function) {
  liveRangeAnalysis = module->getAnalysis<LiveRangeAnalysis>();
  livenessAnalysis = module->getAnalysis<LivenessAnalysis>();
  spillAnalysis = module->getAnalysis<SpillAnalysis>();

  assert(liveRangeAnalysis && livenessAnalysis && spillAnalysis &&
         "LiveRangeAnalysis, LivenessAnalysis, and SpillAnalysis must be "
         "available");

  init();
}

void FunctionTranslater::init() {
  llvm::for_each(
      getTranslateContext()->getRegistersForAllocate(as::RegisterType::Integer),
      [&](as::Register reg) {
        if (reg.isArg())
          intArgRegisters.emplace_back(reg);
      });

  llvm::for_each(getTranslateContext()->getRegistersForAllocate(
                     as::RegisterType::FloatingPoint),
                 [&](as::Register reg) {
                   if (reg.isArg())
                     floatArgRegisters.emplace_back(reg);
                 });

  liveRangeToIndexMap =
      liveRangeAnalysis->getCurrLRIdMap(spillAnalysis->getSpillInfo());

  // initialize `hasCall`
  size_t returnCount = 0;
  function->walk([&](ir::InstructionStorage *inst) -> ir::WalkResult {
    hasCall |= inst->hasTrait<ir::CallLike>();
    returnCount += inst->getDefiningInst<ir::inst::Return>() ? 1 : 0;
    return ir::WalkResult::advance();
  });

  multipleReturn = returnCount > 1;

  if (hasCall) {
    stack.setReturnAddressSize(module->getContext()->getArchitectureByteSize());
    auto sp = stack.returnAddress(0);
    auto anon = createAnonRegister(as::RegisterType::Integer, sp,
                                   as::DataSize::doubleWord(), false);
    returnAddressMemory = anon;
  }

  llvm::SmallVector<LiveRange> spillLiveRange;
  llvm::DenseSet<as::Register> calleeSavedRegisters;
  for (const auto &[liveRange, _] : liveRangeToIndexMap) {
    auto reg =
        irTranslater->getRegisterAllocation().getRegister(function, liveRange);
    if (reg.isCalleeSaved())
      calleeSavedRegisters.insert(reg);

    if (spillAnalysis->getSpillInfo().spilled.contains(liveRange))
      spillLiveRange.emplace_back(liveRange);
  }

  llvm::SmallVector<as::Register> calleeSavedRegistersVec(
      calleeSavedRegisters.begin(), calleeSavedRegisters.end());

  // initialize callee saved registers
  llvm::sort(calleeSavedRegistersVec, [](as::Register a, as::Register b) {
    auto aFloat = a.isFloatingPoint();
    auto bFloat = b.isFloatingPoint();
    if (aFloat != bFloat)
      return aFloat < bFloat;
    return a.getABIIndex() < b.getABIIndex();
  });

  size_t totalCalleeSavedSize = 0;
  for (const auto &reg : calleeSavedRegistersVec) {
    auto sp = stack.calleeSavedRegister(totalCalleeSavedSize);
    totalCalleeSavedSize += module->getContext()->getArchitectureByteSize();

    auto anon = createAnonRegister(as::RegisterType::Integer, sp,
                                   reg.isFloatingPoint()
                                       ? as::DataSize::doublePrecision()
                                       : as::DataSize::doubleWord(),
                                   false);
    calleeSaveInfo.emplace_back(anon, reg);
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

    std::optional<as::DataSize> dataSize;
    if (!type.isa<ir::NameStruct, ir::ArrayT>())
      dataSize = getDataSize(type);

    auto [size, align] = type.getSizeAndAlign(
        module->getOrCreateAnalysis<ir::StructSizeAnalysis>(module)
            ->getStructSizeMap());
    if (size < align)
      size = align;
    auto sp = stack.localVariable(totalLocalVarSize);
    totalLocalVarSize += size;
    auto anon = createAnonRegister(as::RegisterType::Integer, sp, dataSize,
                                   type.isSignedInt());
    localVariablesInfo.emplace_back(anon);

    // initialize localvariable spill
    // It does't need another spill space because local variable memory can be
    // inferred as offset of stack pointer
    auto liveRange = liveRangeAnalysis->getLiveRange(function, localVar);
    if (isSpilled(liveRange)) {
      spillMemories.try_emplace(liveRange, DataSpace::value(anon));
    }
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
    auto dataSize = getDataSize(arg.getType());

    if (arg.getType().isa<ir::FloatT>()) {
      if (argFloatIndex >= getFloatArgRegisters().size()) {
        auto sp = stack.functionArgument(offset);
        offset += module->getContext()->getArchitectureByteSize();
        auto anon =
            createAnonRegister(as::RegisterType::Integer, sp, dataSize, false);
        if (isSpilled(liveRange)) {
          spillMemories.try_emplace(liveRange, DataSpace::memory(anon));
        }

        functionFloatArgMemories.emplace_back(anon);
      }
      argFloatIndex++;
    } else {
      if (argIntIndex >= getIntArgRegisters().size()) {
        auto sp = stack.functionArgument(offset);
        offset += module->getContext()->getArchitectureByteSize();
        auto anon = createAnonRegister(as::RegisterType::Integer, sp, dataSize,
                                       arg.getType().isSignedInt());

        if (isSpilled(liveRange)) {
          spillMemories.try_emplace(liveRange, DataSpace::memory(anon));
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

    auto type = liveRange.getType();
    auto dataSize = getDataSize(type);
    auto sp = stack.spilledRegister(totalSpillSize);
    totalSpillSize += dataSize.getByteSize();
    auto anon = createAnonRegister(as::RegisterType::Integer, sp, dataSize,
                                   type.isSignedInt());
    spillMemories.try_emplace(liveRange, DataSpace::memory(anon));
  }

  stack.setSpilledRegistersSize(totalCalleeSavedSize);
}

as::Register FunctionTranslater::getRegister(ir::Value value) {
  return irTranslater->getRegisterAllocation().getRegister(value);
}

as::Register FunctionTranslater::getRegister(LiveRange liveRange) {
  return irTranslater->getRegisterAllocation().getRegister(function, liveRange);
}

as::Register
FunctionTranslater::getOperandRegister(const ir::Operand *operand) {
  assert(!operand->isConstant() &&
         "Constant operands should be handled separately");
  if (auto it = spillAnalysis->getSpillInfo().restore.find(operand);
      it != spillAnalysis->getSpillInfo().restore.end()) {
    auto restoredLiveRange = it->second;
    return irTranslater->getRegisterAllocation().getRegister(function,
                                                             restoredLiveRange);
  }
  return getRegister(*operand);
}

llvm::SmallVector<as::Register> FunctionTranslater::saveCallerSavedRegisters(
    as::AsmBuilder &builder,
    llvm::ArrayRef<std::pair<as::Register, ir::Type>> datas) {
  size_t totalSize = 0u;

  llvm::SmallVector<std::tuple<StackPoint, as::DataSize, bool>> stackPoint;
  stackPoint.reserve(datas.size());
  for (const auto &[reg, type] : datas) {
    auto dataSize = getDataSize(type);
    stackPoint.emplace_back(stack.callerSavedRegister(totalSize), dataSize,
                            type.isSignedInt());
    totalSize += dataSize.getByteSize();
  }

  stack.setCallerSavedRegistersSize(
      std::max(stack.getCallerSavedRegistersSize(), totalSize));

  llvm::SmallVector<as::Register> anonRegs;
  anonRegs.reserve(datas.size());
  for (const auto &[sp, dataSize, isSigned] : stackPoint) {
    auto anon =
        createAnonRegister(as::RegisterType::Integer, sp, dataSize, isSigned);
    anonRegs.emplace_back(anon);
  }

  if (!datas.empty())
    builder.create<as::CommentLine>("Start of saving caller-saved registers");
  for (size_t i = 0; i < datas.size(); ++i) {
    const auto &[reg, type] = datas[i];
    const auto &anonStackReg = anonRegs[i];
    auto dataSize = std::get<1>(stackPoint[i]);

    storeData(builder, *this, anonStackReg, reg, dataSize, 0);
  }

  if (!datas.empty())
    builder.create<as::CommentLine>("End of saving caller-saved registers");

  return anonRegs;
}

void FunctionTranslater::loadCallerSavedRegisters(
    as::AsmBuilder &builder, llvm::ArrayRef<as::Register> stackpointers,
    llvm::ArrayRef<std::pair<as::Register, ir::Type>> datas) {
  assert(stackpointers.size() == datas.size() &&
         "Stack pointers and data sizes must match in size");

  if (!datas.empty())
    builder.create<as::CommentLine>("Start of restore caller-saved registers");
  for (size_t i = 0; i < stackpointers.size(); ++i) {
    const auto &sp = stackpointers[i];
    const auto &[reg, type] = datas[i];
    auto dataSize = getDataSize(type);

    loadData(builder, *this, reg, sp, dataSize, 0, type.isSignedInt());
  }
  if (!datas.empty())
    builder.create<as::CommentLine>("End of restore caller-saved registers");
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

  RegisterAllocation &regAlloc =
      translater.getIRTranslater()->getRegisterAllocation();
  LiveRangeAnalysis *liveRangeAnalysis = translater.getLiveRangeAnalysis();

  llvm::DenseSet<LiveRange> liveIn = callLiveness->getLiveIn(inst);
  llvm::SmallVector<std::pair<as::Register, ir::Type>, 16> toSave;
  for (LiveRange lr : liveIn) {
    auto reg = regAlloc.getRegister(translater.getFunction(), lr);
    if (reg.isCallerSaved()) {
      auto type = lr.getType();
      toSave.emplace_back(reg, type);
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
      if (translater.getSpillAnalysis()->getSpillInfo().restore.contains(
              &operand))
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
  as::AsmBuilder::InsertionGuard guard(builder);
  auto newBlock = translater.createBlock(block);
  builder.setInsertionPointStart(newBlock);

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

as::Function *FunctionTranslater::translate() {
  llvm::SmallVector<as::Block *> blocks;
  blocks.reserve(function->getBlockCount() + 1);
  auto funcEntryBlock = new as::Block(function->getName());
  blocks.emplace_back(funcEntryBlock);

  as::AsmBuilder builder;
  for (ir::Block *block : *function) {
    auto asBlock = translateBlock(builder, *this, block);
    if (!asBlock) {
      for (auto *b : blocks)
        delete b;
      return nullptr;
    }
    blocks.emplace_back(asBlock);
  }

  builder.setInsertionPointStart(funcEntryBlock);
  writeFunctionStart(builder);

  if (multipleReturn) {
    auto endBlock = new as::Block(functionEndLabel());
    builder.setInsertionPointStart(endBlock);
    writeFunctionEndImpl(builder);
    blocks.emplace_back(endBlock);
  }

  as::Function *newFunction = new as::Function(blocks);

  substitueAnonymousRegisters(newFunction);
  return newFunction;
}

void FunctionTranslater::writeFunctionStart(as::AsmBuilder &builder) {
  auto tempReg = getTranslateContext()->getTempRegisters()[0];

  if (auto totalSize = stack.getTotalSize()) {
    auto *imm =
        getImmOrLoad(builder, tempReg, -static_cast<std::int32_t>(totalSize));
    if (imm) {
      builder.create<as::itype::Addi>(as::Register::sp(), as::Register::sp(),
                                      imm, as::DataSize::doubleWord());
    } else {
      builder.create<as::rtype::Add>(as::Register::sp(), as::Register::sp(),
                                     tempReg, as::DataSize::doubleWord());
    }
  }

  if (stack.getReturnAddressSize() > 0) {
    auto sp = stack.returnAddress(0);
    auto anon = createAnonRegister(as::RegisterType::Integer, sp,
                                   as::DataSize::doubleWord(), false);
    builder.create<as::stype::Store>(anon, as::Register::ra(), getImmediate(0),
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
        builder.create<as::itype::Load>(rd, sp, getImmediate(0), dataSize,
                                        true);
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
      bool isSigned = false;
      if (auto intT = arg.getType().dyn_cast<ir::IntT>()) {
        isSigned = intT.isSigned();
      }

      if (argIntIndex >= getIntArgRegisters().size()) {
        auto sp =
            functionIntArgMemories[argIntIndex - getIntArgRegisters().size()];
        auto load = builder.create<as::itype::Load>(rd, sp, getImmediate(0),
                                                    dataSize, isSigned);
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
  if (multipleReturn)
    builder.create<as::pseudo::J>(functionEndLabel());
  else
    writeFunctionEndImpl(builder);
}

void FunctionTranslater::writeFunctionEndImpl(as::AsmBuilder &builder) {
  // Restore callee saved registers
  loadCalleeSavedRegisters(builder);

  // Restore return address
  if (stack.getReturnAddressSize() > 0) {
    auto sp = *returnAddressMemory;
    builder.create<as::itype::Load>(as::Register::ra(), sp, getImmediate(0),
                                    as::DataSize::doubleWord(), true);
  }

  // stack cleanup
  if (stack.getTotalSize() > 0) {
    auto tempReg = getTranslateContext()->getTempRegisters()[0];
    auto *imm = getImmOrLoad(builder, tempReg, stack.getTotalSize());
    if (imm) {
      builder.create<as::itype::Addi>(as::Register::sp(), as::Register::sp(),
                                      imm, as::DataSize::doubleWord());
    } else {
      builder.create<as::rtype::Add>(as::Register::sp(), as::Register::sp(),
                                     tempReg, as::DataSize::doubleWord());
    }
  }

  builder.create<as::pseudo::Ret>();
}

std::string FunctionTranslater::functionEndLabel() const {
  return std::format(".{}_Lend", function->getName().str());
}

void FunctionTranslater::moveRegisters(as::AsmBuilder &builder,
                                       llvm::ArrayRef<DataSpace> srcs,
                                       llvm::ArrayRef<DataSpace> dsts) {
  assert(srcs.size() == dsts.size() &&
         "Source and destination sizes must match");
  auto tempReg0 = getTranslateContext()->getTempRegisters()[0];
  auto scheduler =
      MoveManagement<DataSpace>(dsts, srcs, DataSpace::value(tempReg0));
  auto moveSchedule = scheduler.getMoveSchedule();

  for (auto [movement, dstData, srcData] : moveSchedule) {
    auto dst = dstData.getRegister();
    auto src = srcData.getRegister();
    auto dataSize = src.isFloatingPoint() ? as::DataSize::doublePrecision()
                                          : as::DataSize::doubleWord();

    switch (movement) {
    case Movement::Move: {
      if (dstData.isMemory() && !srcData.isMemory()) {
        // store to dst
        builder.create<as::stype::Store>(dst, src, getImmediate(0), dataSize);
      } else if (srcData.isMemory()) {
        // load from src
        if (dstData.isMemory()) {
          // load from src to temp, then store to dst
          builder.create<as::itype::Load>(tempReg0, src, getImmediate(0),
                                          as::DataSize::doubleWord(), false);
          builder.create<as::stype::Store>(dst, tempReg0, getImmediate(0),
                                           as::DataSize::doubleWord());
        } else {
          // load from src to dst
          builder.create<as::itype::Load>(dst, src, getImmediate(0), dataSize,
                                          false);
        }
      } else {
        // register to register move
        builder.create<as::pseudo::Mv>(dst, src);
      }
      break;
    }
    case Movement::Swap: {
      if (!dstData.isMemory() && !srcData.isMemory()) {
        // swap(a, b) ->
        //    a = a ^ b;
        //    b = a ^ b;
        //    a = a ^ b;
        builder.create<as::rtype::Xor>(dst, dst, src);
        builder.create<as::rtype::Xor>(src, dst, src);
        builder.create<as::rtype::Xor>(dst, dst, src);
      } else if (dstData.isMemory() != srcData.isMemory()) {
        if (dstData.isMemory())
          std::swap(dst, src);
        // dst is register, src is memory
        dataSize = dst.isFloatingPoint() ? as::DataSize::doublePrecision()
                                         : as::DataSize::doubleWord();

        // swap(a, mem) ->
        //   temp = a
        //   load a from mem
        //   store temp to mem

        if (dst.isFloatingPoint())
          builder.create<as::rtype::FmvFloatToInt>(
              tempReg0, dst, std::nullopt, as::DataSize::doublePrecision());
        else
          builder.create<as::pseudo::Mv>(tempReg0, dst);

        builder.create<as::itype::Load>(dst, src, getImmediate(0), dataSize,
                                        false);

        builder.create<as::stype::Store>(src, tempReg0, getImmediate(0),
                                         dataSize);
      } else {
        // both are memory
        // swap(mem1, mem2) ->
        // temp1 = load mem1
        // temp2 = load mem2
        // store temp1 to mem2
        // store temp2 to mem1
        auto tempReg1 = getTranslateContext()->getTempRegisters()[1];
        builder.create<as::itype::Load>(tempReg0, dst, getImmediate(0),
                                        as::DataSize::doubleWord(), false);
        builder.create<as::itype::Load>(tempReg1, src, getImmediate(0),
                                        as::DataSize::doubleWord(), false);
        builder.create<as::stype::Store>(src, tempReg0, getImmediate(0),
                                         as::DataSize::doubleWord());
        builder.create<as::stype::Store>(dst, tempReg1, getImmediate(0),
                                         as::DataSize::doubleWord());
      }
      break;
    }
    }
  }
}

void FunctionTranslater::spillRegister(as::AsmBuilder &builder,
                                       LiveRange liveRange) {
  auto memory = spillMemories.at(liveRange);

  if (memory.isMemory()) {
    auto type = liveRange.getType();
    auto dataSize = getDataSize(type);
    auto reg =
        irTranslater->getRegisterAllocation().getRegister(function, liveRange);

    auto store = builder.create<as::stype::Store>(memory.getRegister(), reg,
                                                  getImmediate(0), dataSize);
    store->setComment(std::format("{}bytes spill", dataSize.getByteSize()));
  }
}

bool FunctionTranslater::isSpilled(LiveRange liveRange) const {
  return spillAnalysis->getSpillInfo().spilled.contains(liveRange);
}

as::Register FunctionTranslater::restoreOperand(as::AsmBuilder &builder,
                                                const ir::Operand *operand) {
  auto restoredLiveRange = spillAnalysis->getSpillInfo().restore.at(operand);
  auto spilledLiveRange = liveRangeAnalysis->getLiveRange(function, *operand);
  assert(!isSpilled(restoredLiveRange) && "Restored value must not be spilled");

  auto rd = irTranslater->getRegisterAllocation().getRegister(
      function, restoredLiveRange);

  auto spilledMemory = spillMemories.at(spilledLiveRange);
  if (spilledMemory.isMemory()) {
    auto type = operand->getType();
    auto dataSize = getDataSize(type);
    bool isSigned = false;
    if (auto intT = type.dyn_cast<ir::IntT>())
      isSigned = intT.isSigned();

    auto load = builder.create<as::itype::Load>(
        rd, spilledMemory.getRegister(), getImmediate(0), dataSize, isSigned);
    load->setComment(std::format("{}bytes restore", dataSize.getByteSize()));
  } else {
    auto sp = spilledMemory.getRegister();
    auto mv = builder.create<as::pseudo::Mv>(rd, sp);
    mv->setComment("local variable");
  }
  return rd;
}

void FunctionTranslater::saveCalleeSavedRegisters(as::AsmBuilder &builder) {
  for (const auto &[sp, reg] : calleeSaveInfo) {
    auto dataSize = reg.isFloatingPoint() ? as::DataSize::doublePrecision()
                                          : as::DataSize::doubleWord();
    builder.create<as::stype::Store>(sp, reg, getImmediate(0), dataSize);
  }
}

void FunctionTranslater::loadCalleeSavedRegisters(as::AsmBuilder &builder) {
  for (const auto &[sp, reg] : calleeSaveInfo) {
    auto dataSize = reg.isFloatingPoint() ? as::DataSize::doublePrecision()
                                          : as::DataSize::doubleWord();
    builder.create<as::itype::Load>(reg, sp, getImmediate(0), dataSize, true);
  }
}

as::Register FunctionTranslater::createAnonRegister(
    as::RegisterType regType, const StackPoint &sp,
    std::optional<as::DataSize> dataSize, bool isSigned) {
  auto anon = as::Register::createAnonymousRegister(
      getTranslateContext()->getAnonymousRegStorage(), regType,
      as::CallingConvension::None, getAnonRegLabel());
  anonymousRegisterToSp.try_emplace(anon, sp, dataSize, isSigned);
  return anon;
}

void FunctionTranslater::substitueAnonymousRegisters(as::Function *function) {
  llvm::SmallVector<as::Instruction *> toSubst;

  function->walk([&](as::Instruction *inst) {
    if (auto *mv = inst->dyn_cast<as::pseudo::Mv>()) {
      if (mv->getRs().isAnonymous())
        toSubst.emplace_back(inst);
    } else if (auto *load = inst->dyn_cast<as::itype::Load>()) {
      if (load->getBase().isAnonymous()) {
        assert(load->getOffset()->isZero());
        toSubst.emplace_back(inst);
      }
    } else if (auto *store = inst->dyn_cast<as::stype::Store>()) {
      if (store->getBase().isAnonymous()) {
        assert(store->getOffset()->isZero());
        toSubst.emplace_back(inst);
      }
    }
  });

  as::AsmBuilder builder;
  auto tempReg = getTranslateContext()->getTempRegisters()[0];
  for (as::Instruction *inst : toSubst) {
    as::AsmBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointBeforeInst(inst);

    llvm::StringRef comment = inst->getComment();
    if (auto *mv = inst->dyn_cast<as::pseudo::Mv>()) {
      auto [sp, dataSize, isSigned] = anonymousRegisterToSp.at(mv->getRs());
      auto offset = sp.fromBottom();

      auto imm = getImmOrLoad(builder, tempReg, offset);
      if (imm) {
        builder
            .create<as::itype::Addi>(mv->getRd(), as::Register::sp(), imm,
                                     as::DataSize::doubleWord())
            ->setComment(comment);
      } else {
        builder
            .create<as::rtype::Add>(mv->getRd(), as::Register::sp(), tempReg,
                                    as::DataSize::doubleWord())
            ->setComment(comment);
      }
    } else if (auto *load = inst->dyn_cast<as::itype::Load>()) {
      auto [sp, dataSize, isSigned] = anonymousRegisterToSp.at(load->getBase());
      assert(dataSize.has_value() && "Data size must be known");
      auto offset = sp.fromBottom();

      auto imm = getImmOrLoad(builder, tempReg, offset);
      if (imm) {
        builder
            .create<as::itype::Load>(load->getDst(), as::Register::sp(), imm,
                                     *dataSize, isSigned)
            ->setComment(comment);
      } else {
        builder.create<as::rtype::Add>(tempReg, as::Register::sp(), tempReg,
                                       as::DataSize::doubleWord());
        builder
            .create<as::itype::Load>(load->getDst(), tempReg, getImmediate(0),
                                     *dataSize, isSigned)
            ->setComment(comment);
      }
    } else /* store */ {
      auto *store = inst->cast<as::stype::Store>();
      auto [sp, dataSize, isSigned] =
          anonymousRegisterToSp.at(store->getBase());

      assert(dataSize.has_value() && "Data size must be known");
      auto offset = sp.fromBottom();
      auto imm = getImmOrLoad(builder, tempReg, offset);
      if (imm) {
        builder
            .create<as::stype::Store>(as::Register::sp(), store->getSrc(), imm,
                                      *dataSize)
            ->setComment(comment);
      } else {
        builder.create<as::rtype::Add>(tempReg, as::Register::sp(), tempReg,
                                       as::DataSize::doubleWord());
        builder
            .create<as::stype::Store>(tempReg, store->getSrc(), getImmediate(0),
                                      *dataSize)
            ->setComment(comment);
      }
    }

    inst->remove();
  }
}

TranslationRuleSet::TranslationRuleSet() = default;
TranslationRuleSet::~TranslationRuleSet() = default;

void TranslationRuleSet::addRule(TypeID id,
                                 std::unique_ptr<TranslationRule> rule) {
  auto [it, inserted] = rules.try_emplace(id, std::move(rule));
  assert(inserted && "Rule for this TypeID already exists");
  (void)inserted;
}

TranslationRule *TranslationRuleSet::getRule(TypeID id) const {
  auto it = rules.find(id);
  if (it != rules.end())
    return it->second.get();
  return nullptr;
}

as::Immediate *getImmediate(int64_t value) {
  return new as::ValueImmediate(value);
}
as::Immediate *getImmediate(ir::ConstantAttr constAttr) {
  auto intConst = constAttr.dyn_cast<ir::ConstantIntAttr>();
  assert(intConst && "Only ConstantIntAttr is supported");
  return getImmediate(intConst.getValue());
}
as::Immediate *getImmediate(ir::inst::Constant constant) {
  return getImmediate(constant.getValue());
}

static constexpr std::int32_t HI12 = (1 << 11) - 1;
static constexpr std::int32_t LO12 = -(1 << 11);

void loadInt32(as::AsmBuilder &builder, as::Register rd, std::int32_t value) {
  if (LO12 <= value && value <= HI12) {
    builder.create<as::itype::Addi>(rd, as::Register::zero(),
                                    getImmediate(value), as::DataSize::word());
  } else {
    std::uint32_t hi12 = (static_cast<std::uint32_t>(value) >> 12u);
    std::int32_t lo12 = (value << 20) >> 20;
    if (lo12 < 0) {
      hi12 += 1;
    }

    builder.create<as::utype::Lui>(rd, getImmediate(hi12));
    if (lo12 != 0) {
      builder.create<as::itype::Addi>(rd, rd, getImmediate(lo12),
                                      as::DataSize::word());
    }
  }
}

void loadInt64(as::AsmBuilder &builder, FunctionTranslater &translater,
               as::Register rd, std::int64_t value) {
  // If the low bits are zero, shift the whole value to the right, treat it as a
  // small int and load it, then shift it back to the left.

  std::int64_t shiftedValue = value;
  size_t shiftRight = 0;
  while (!(shiftedValue & 1)) {
    shiftedValue >>= 1;
    shiftRight++;
  }

  if (static_cast<std::int32_t>(shiftedValue) == shiftedValue) {
    // If the value can fit in a 32-bit signed integer, treat it as a small int.
    loadInt32(builder, rd, static_cast<std::int32_t>(shiftedValue));

    if (shiftRight > 0) {
      builder.create<as::itype::Slli>(rd, rd, getImmediate(shiftRight),
                                      as::DataSize::doubleWord());
    }
  } else {
    auto [label, dataSize] = translater.getConstantLabel(value);
    builder.create<as::pseudo::La>(rd, label);
    builder.create<as::itype::Load>(rd, rd, getImmediate(0), *dataSize, true);
  }
}

void loadInt(as::AsmBuilder &builder, FunctionTranslater &translater,
             as::Register rd, std::int64_t value, ir::IntT intT) {
  if (intT.getBitWidth() <= 32 || (static_cast<std::int32_t>(value) == value)) {
    // If the value can fit in a 32-bit signed integer, treat it as a small int.
    loadInt32(builder, rd, static_cast<std::int32_t>(value));
  } else {
    // Otherwise, treat it as a big int.
    loadInt64(builder, translater, rd, value);
  }
}

void loadFloat(as::AsmBuilder &builder, FunctionTranslater &translater,
               as::Register rd, llvm::APFloat value) {
  bool isSinglePrecision =
      (&value.getSemantics() == &llvm::APFloat::IEEEsingle());

  auto tempReg = translater.getTranslateContext()->getTempRegisters()[0];
  auto intValue = value.bitcastToAPInt().getZExtValue();
  if (isSinglePrecision) {
    if (intValue == 0) {
      builder.create<as::rtype::FmvIntToFloat>(rd, as::Register::zero(),
                                               std::nullopt,
                                               as::DataSize::singlePrecision());
      return;
    }
    loadInt32(builder, tempReg, static_cast<std::int32_t>(intValue));
    builder.create<as::rtype::FmvIntToFloat>(rd, tempReg, std::nullopt,
                                             as::DataSize::singlePrecision());
  } else {
    if (intValue == 0) {
      builder.create<as::rtype::FmvIntToFloat>(rd, as::Register::zero(),
                                               std::nullopt,
                                               as::DataSize::doublePrecision());
      return;
    }
    // If the low bits are zero, shift the whole value to the right, treat it as
    // a
    // small int and load it, then shift it back to the left.

    std::int64_t shiftedValue = intValue;
    size_t shiftRight = 0;
    while (!(shiftedValue & 1)) {
      shiftedValue >>= 1;
      shiftRight++;
    }

    if (static_cast<std::int32_t>(shiftedValue) == shiftedValue) {
      // If the value can fit in a 32-bit signed integer, treat it as a small
      // int.
      loadInt32(builder, tempReg, static_cast<std::int32_t>(shiftedValue));

      if (shiftRight > 0) {
        builder.create<as::itype::Slli>(tempReg, tempReg,
                                        getImmediate(shiftRight),
                                        as::DataSize::doubleWord());
      }
      builder.create<as::rtype::FmvIntToFloat>(rd, tempReg, std::nullopt,
                                               as::DataSize::doublePrecision());
    } else {
      auto [label, dataSize] =
          translater.getConstantLabel(ir::ConstantFloatAttr::get(
              translater.getModule()->getContext(), value));
      builder.create<as::pseudo::La>(tempReg, label);
      builder.create<as::itype::Load>(rd, tempReg, getImmediate(0), *dataSize,
                                      false);
    }
  }
}

as::Immediate *getImmOrLoad(as::AsmBuilder &builder, as::Register rd,
                            std::int32_t value) {

  if (LO12 <= value && value <= HI12) {
    return getImmediate(value);
  } else {
    loadInt32(builder, rd, value);
    return nullptr; // indicates that a load was created
  }
}

void storeData(as::AsmBuilder &builder, FunctionTranslater &translater,
               as::Register rd, as::Register rs, as::DataSize dataSize,
               std::int32_t offset) {
  auto tempReg = translater.getTranslateContext()->getTempRegisters()[0];
  auto *imm = getImmOrLoad(builder, tempReg, offset);
  if (imm) {
    builder.create<as::stype::Store>(rd, rs, imm, dataSize);
  } else {
    builder.create<as::rtype::Add>(rd, rd, tempReg, as::DataSize::doubleWord());
    builder.create<as::stype::Store>(rd, rs, getImmediate(0), dataSize);
  }
}

void loadData(as::AsmBuilder &builder, FunctionTranslater &translater,
              as::Register rd, as::Register rs, as::DataSize dataSize,
              std::int32_t offset, bool isSigned) {
  auto tempReg = translater.getTranslateContext()->getTempRegisters()[0];
  auto *imm = getImmOrLoad(builder, tempReg, offset);
  if (imm) {
    builder.create<as::itype::Load>(rd, rs, imm, dataSize, true);
  }
}

as::DataSize getDataSize(ir::Type type) {
  auto [size, _] = type.getSizeAndAlign(ir::StructSizeMap{});
  if (type.isa<ir::FloatT>()) {
    if (size == 4) {
      return as::DataSize::singlePrecision();
    } else if (size == 8) {
      return as::DataSize::doublePrecision();
    }
  } else {
    if (size <= 1) {
      return as::DataSize::byte();
    } else if (size == 2) {
      return as::DataSize::half();
    } else if (size == 4) {
      return as::DataSize::word();
    } else if (size == 8) {
      return as::DataSize::doubleWord();
    }
  }
  llvm::report_fatal_error(llvm::StringRef(std::format(
      "Unsupported type to convert to data size: {}", type.toString())));
}

} // namespace kecc
