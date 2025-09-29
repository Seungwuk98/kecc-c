#include "kecc/ir/Interpreter.h"
#include "kecc/ir/IRAnalyses.h"
#include "kecc/ir/IRAttributes.h"
#include "kecc/ir/IRInstructions.h"
#include "kecc/ir/IRTypes.h"
#include "kecc/ir/Instruction.h"
#include "kecc/ir/TypeAttributeSupport.h"
#include "kecc/ir/WalkSupport.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MemAlloc.h"
#include "llvm/Support/SourceMgr.h"

namespace kecc::ir {

static std::unique_ptr<VRegister>
createVRegister(ir::Type type, const StructSizeMap &sizeMap) {
  if (type.isa<ir::FloatT>()) {
    return std::make_unique<VRegisterFloat>();
  } else if (auto structT = type.dyn_cast<NameStruct>()) {
    auto [size, align] = structT.getSizeAndAlign(sizeMap);
    if (size < align)
      size = align;
    return std::make_unique<VRegisterDynamic>(size);
  } else {
    return std::make_unique<VRegisterInt>();
  }
}

static void printDynamicImpl(llvm::raw_ostream &os, void *mem, Type type,
                             StructSizeAnalysis *analysis) {
  os << type << '(';
  llvm::TypeSwitch<Type>(type)
      .Case([&](IntT intT) {
        if (intT.getBitWidth() <= 8) {
          auto value = *static_cast<std::uint8_t *>(mem);
          if (intT.getBitWidth() == 1)
            os << (value ? "true" : "false");
          else if (intT.isSigned()) {
            os << static_cast<int>(static_cast<int8_t>(value));
          } else {
            os << static_cast<int>(static_cast<uint8_t>(value));
          }
        } else if (intT.getBitWidth() <= 16) {
          auto value = *static_cast<std::uint16_t *>(mem);
          if (intT.isSigned()) {
            os << static_cast<int16_t>(value);
          } else {
            os << static_cast<uint16_t>(value);
          }
        } else if (intT.getBitWidth() <= 32) {
          auto value = *static_cast<std::uint32_t *>(mem);
          if (intT.isSigned()) {
            os << static_cast<int32_t>(value);
          } else {
            os << static_cast<uint32_t>(value);
          }
        } else if (intT.getBitWidth() <= 64) {
          auto value = *static_cast<std::uint64_t *>(mem);
          if (intT.isSigned()) {
            os << static_cast<int64_t>(value);
          } else {
            os << static_cast<uint64_t>(value);
          }
        } else {
          llvm_unreachable("Unsupported integer bit width");
        }
      })

      .Case([&](FloatT floatT) {
        llvm::APFloat apFloat = [&]() {
          if (floatT.getBitWidth() == 32) {
            std::uint32_t value = *static_cast<std::uint32_t *>(mem);
            llvm::APInt apInt(32, value, false);
            return llvm::APFloat(llvm::APFloat::IEEEsingle(), apInt);
          } else {
            std::uint64_t value = *static_cast<std::uint64_t *>(mem);
            llvm::APInt apInt(64, value, false);
            return llvm::APFloat(llvm::APFloat::IEEEdouble(), apInt);
          }
        }();
        llvm::SmallVector<char> buffer;
        apFloat.toString(buffer);
        os << llvm::StringRef(buffer.data(), buffer.size());
      })

      .Case([&](PointerT pointerT) {
        auto ptr = *static_cast<std::uint64_t *>(mem);
        os << llvm::format_hex(ptr, 18);
      })

      .Case([&](NameStruct structT) {
        const auto &[size, align, offsets] =
            analysis->getStructSizeMap().at(structT.getName());
        const auto &fieldTypes =
            analysis->getStructFieldsMap().at(structT.getName());
        os << '{';
        for (const auto &[offset, fieldType] : llvm::zip(offsets, fieldTypes)) {
          if (offset != 0)
            os << ", ";
          printDynamicImpl(os, static_cast<char *>(mem) + offset, fieldType,
                           analysis);
        }
        os << '}';
      })

      .Case([&](ArrayT arrayT) {
        auto elemType = arrayT.getElementType();
        auto [size, align] =
            elemType.getSizeAndAlign(analysis->getStructSizeMap());
        if (size < align)
          size = align;

        auto arraySize = arrayT.getSize();
        os << '{';
        for (size_t i = 0; i < arraySize; ++i) {
          if (i != 0)
            os << ", ";
          printDynamicImpl(os, static_cast<char *>(mem) + i * size, elemType,
                           analysis);
        }
        os << '}';
      })
      .Default([&](Type) {
        llvm_unreachable("Unsupported type for struct printing");
      });
  os << ')';
}

VMemory VMemory::getElementPtr(int offset) const {
  assert(data && "Memory pointer cannot be null");
  return VMemory(static_cast<char *>(data) + offset);
}

void VMemory::loadInto(VRegister *dest, Type type,
                       StructSizeAnalysis *analysis) const {
  assert(data && "Memory pointer cannot be null");
  llvm::TypeSwitch<Type>(type)
      .Case([&](IntT intT) {
        auto bitWidth = intT.getBitWidth();
        auto isSigned = intT.isSigned();
        if (bitWidth == 1)
          isSigned = false;

        std::uint64_t value = 0;
        VRegisterInt *destIntR = dest->cast<VRegisterInt>();
        if (bitWidth <= 8) {
          value = *static_cast<std::uint8_t *>(data);
        } else if (bitWidth == 16) {
          value = *static_cast<std::uint16_t *>(data);
        } else if (bitWidth == 32) {
          value = *static_cast<std::uint32_t *>(data);
        } else if (bitWidth == 64) {
          value = *static_cast<std::uint64_t *>(data);
        } else {
          llvm_unreachable("Unsupported integer bit width");
        }
        llvm::APInt apInt(bitWidth, value, isSigned);
        llvm::APSInt apSInt(apInt, !isSigned);
        destIntR->setValue(apSInt);
      })

      .Case([&](FloatT floatT) {
        auto bitWidth = floatT.getBitWidth();
        std::uint64_t value = 0;
        VRegisterFloat *destFloatR = dest->cast<VRegisterFloat>();
        if (bitWidth == 32) {
          value = *static_cast<std::uint32_t *>(data);
        } else if (bitWidth == 64) {
          value = *static_cast<std::uint64_t *>(data);
        } else {
          llvm_unreachable("Unsupported float bit width");
        }
        destFloatR->setValue(value);
      })

      .Case([&](PointerT pointerT) {
        auto ptr = *static_cast<std::uint64_t *>(data);
        VRegisterInt *destIntR = dest->cast<VRegisterInt>();
        destIntR->setValue(ptr);
      })

      .Case([&](ArrayT arrayT) {
        const auto &[size, align] =
            arrayT.getSizeAndAlign(analysis->getStructSizeMap());
        auto *destDynR = dest->cast<VRegisterDynamic>();
        assert(destDynR->getSize() == size &&
               "Destination dynamic register must have the same size");
        std::memcpy(destDynR->getData(), data, size);
      })

      .Case([&](NameStruct structT) {
        const auto &[size, align] =
            structT.getSizeAndAlign(analysis->getStructSizeMap());
        auto *destDynR = dest->cast<VRegisterDynamic>();
        assert(destDynR->getSize() == size &&
               "Destination dynamic register must have the same size");
        std::memcpy(destDynR->getData(), data, size);
      })

      .Default([&](Type) {
        llvm_unreachable("Unsupported type for load instruction");
      });
}

void VMemory::storeFrom(VRegister *src, Type type,
                        StructSizeAnalysis *analysis) {
  assert(data && "Memory pointer cannot be null");
  llvm::TypeSwitch<Type>(type)
      .Case([&](IntT intT) {
        auto bitWidth = intT.getBitWidth();
        auto isSigned = intT.isSigned();
        if (bitWidth == 1)
          isSigned = false;

        VRegisterInt *srcIntR = src->cast<VRegisterInt>();
        llvm::APSInt apInt = srcIntR->getAsInteger(bitWidth, isSigned);
        if (bitWidth <= 8) {
          *static_cast<std::uint8_t *>(data) =
              static_cast<std::uint8_t>(apInt.getZExtValue());
        } else if (bitWidth == 16) {
          *static_cast<std::uint16_t *>(data) =
              static_cast<std::uint16_t>(apInt.getZExtValue());
        } else if (bitWidth == 32) {
          *static_cast<std::uint32_t *>(data) =
              static_cast<std::uint32_t>(apInt.getZExtValue());
        } else if (bitWidth == 64) {
          *static_cast<std::uint64_t *>(data) =
              static_cast<std::uint64_t>(apInt.getZExtValue());
        } else {
          llvm_unreachable("Unsupported integer bit width");
        }
      })

      .Case([&](FloatT floatT) {
        auto bitWidth = floatT.getBitWidth();
        VRegisterFloat *srcFloatR = src->cast<VRegisterFloat>();
        auto value = srcFloatR->getValue();
        if (bitWidth == 32) {
          *static_cast<std::uint32_t *>(data) =
              static_cast<std::uint32_t>(value);
        } else if (bitWidth == 64) {
          *static_cast<std::uint64_t *>(data) =
              static_cast<std::uint64_t>(value);
        } else {
          llvm_unreachable("Unsupported float bit width");
        }
      })

      .Case([&](PointerT pointerT) {
        VRegisterInt *srcIntR = src->cast<VRegisterInt>();
        auto ptr = srcIntR->getValue();
        *static_cast<std::uint64_t *>(data) = ptr;
      })

      .Case([&](ArrayT arrayT) {
        const auto &[size, align] =
            arrayT.getSizeAndAlign(analysis->getStructSizeMap());
        auto *srcDynR = src->cast<VRegisterDynamic>();
        assert(srcDynR->getSize() == size &&
               "Source dynamic register must have the same size");
        std::memcpy(data, srcDynR->getData(), size);
      })

      .Case([&](NameStruct structT) {
        const auto &[size, align] =
            structT.getSizeAndAlign(analysis->getStructSizeMap());
        auto *srcDynR = src->cast<VRegisterDynamic>();
        assert(srcDynR->getSize() == size &&
               "Source dynamic register must have the same size");
        std::memcpy(data, srcDynR->getData(), size);
      })

      .Default([&](Type) {
        llvm_unreachable("Unsupported type for store instruction");
      });
}

void VMemory::print(llvm::raw_ostream &os, Type type,
                    StructSizeAnalysis *analysis) const {
  assert(data && "Memory pointer cannot be null");
  printDynamicImpl(os, data, type, analysis);
}

void VRegisterInt::print(llvm::raw_ostream &os, Type type,
                         StructSizeAnalysis *analysis) const {
  os << "Raw value: " << value << " type: " << type;
  if (type.isa<ir::PointerT>()) {
    os << ' ' << llvm::format_hex(value, 18);
  } else if (auto intT = type.dyn_cast<ir::IntT>()) {
    bool isSigned = intT.isSigned();
    if (intT.getBitWidth() == 1)
      isSigned = false;
    auto apInt = getAsInteger(intT.getBitWidth(), isSigned);
    os << ' ' << apInt;
  }
}

void VRegisterInt::mv(VRegister *src) {
  assert(src->isa<VRegisterInt>() && "Source must be an integer register");
  VRegisterInt *srcIntR = src->cast<VRegisterInt>();
  setValue(srcIntR->getValue());
}

std::unique_ptr<VRegister> VRegisterInt::clone() const {
  return std::make_unique<VRegisterInt>(value);
}

llvm::APSInt VRegisterInt::getAsInteger(unsigned bitWidth,
                                        bool isSigned) const {
  llvm::APInt apInt(bitWidth, value, isSigned);
  return llvm::APSInt(apInt, !isSigned);
}
VMemory VRegisterInt::getAsMemory() const {
  return VMemory(reinterpret_cast<void *>(value));
}

void VRegisterInt::setValue(llvm::APSInt v) {
  bool isSigned = v.isSigned();
  if (v.getBitWidth() == 1)
    isSigned = false;

  value = v.isSigned() ? v.getSExtValue() : v.getZExtValue();
}

void VRegisterInt::setValue(VMemory v) {
  auto ptr = v.getData();
  assert(ptr && "Memory pointer cannot be null");
  value = reinterpret_cast<std::uint64_t>(ptr);
}

void VRegisterFloat::print(llvm::raw_ostream &os, Type type,
                           StructSizeAnalysis *analysis) const {
  assert(type.isa<ir::FloatT>() && "Type must be a float type");
  os << "Raw value: " << value << " type: " << type;
  auto floatT = type.cast<ir::FloatT>();
  auto apFloat = getAsFloat(floatT.getBitWidth());

  llvm::SmallVector<char> buffer;
  apFloat.toString(buffer);
  os << ' ' << llvm::StringRef(buffer.data(), buffer.size());
}

void VRegisterFloat::mv(VRegister *src) {
  assert(src->isa<VRegisterFloat>() && "Source must be a float register");
  VRegisterFloat *srcFloatR = src->cast<VRegisterFloat>();
  setValue(srcFloatR->getValue());
}

std::unique_ptr<VRegister> VRegisterFloat::clone() const {
  return std::make_unique<VRegisterFloat>(value);
}

void VRegisterFloat::setValue(llvm::APFloat v) {
  value = v.bitcastToAPInt().getZExtValue();
}
void VRegisterFloat::setValue(std::uint64_t v) { value = v; }

llvm::APFloat VRegisterFloat::getAsFloat(int bitwidth) const {
  if (bitwidth == 32) {
    llvm::APInt apInt(32, value, false);
    return llvm::APFloat(llvm::APFloat::IEEEsingle(), apInt);
  } else {
    llvm::APInt apInt(64, value, false);
    return llvm::APFloat(llvm::APFloat::IEEEdouble(), apInt);
  }
}

void VRegisterDynamic::print(llvm::raw_ostream &os, Type type,
                             StructSizeAnalysis *analysis) const {
  assert(type.isa<NameStruct>() && "Type must be a struct type");
  os << "Dynamic size: " << size << " raw value: ";

  llvm::SmallVector<std::uint8_t> values;
  values.resize(size);
  for (size_t i = 0; i < size; ++i) {
    values[i] = static_cast<std::uint8_t>(static_cast<char *>(data)[i]);
  }

  std::reverse(values.begin(), values.end());

  size_t i = 0;
  for (; i < size % 4; ++i) {
    os << llvm::format_hex_no_prefix(values[i], 2);
  }
  if (size % 4)
    os << ' ';
  size_t count = 0;
  for (; i < size; ++i) {
    os << llvm::format_hex_no_prefix(values[i], 2);
    if (++count == 4 && i + 1 < size) {
      os << ' ';
      count = 0;
    }
  }

  os << " type: " << type << ' ';

  printDynamicImpl(os, getData(), type, analysis);
}

void VRegisterDynamic::mv(VRegister *src) {
  assert(src->isa<VRegisterDynamic>() && "Source must be a dynamic register");
  VRegisterDynamic *srcDynR = src->cast<VRegisterDynamic>();
  assert(srcDynR->getSize() == getSize() &&
         "Source and destination must have the same size");
  std::memcpy(data, srcDynR->getData(), getSize());
}

std::unique_ptr<VRegister> VRegisterDynamic::clone() const {
  auto newReg = std::make_unique<VRegisterDynamic>(size);
  std::memcpy(newReg->getData(), data, size);
  return newReg;
}

StackFrame::~StackFrame() {
  if (frameMem)
    free(frameMem);
}

void StackFrame::initRegisters() {
  const auto &structSizeMap =
      interpreter->structSizeAnalysis->getStructSizeMap();

  frameSize = 0;
  llvm::SmallVector<size_t> offsets;

  for (InstructionStorage *inst : *function->getAllocationBlock()) {
    inst::LocalVariable localVar = inst->getDefiningInst<inst::LocalVariable>();
    assert(localVar && "Allocation block can only contain local variable");

    ir::Type type = localVar.getType().cast<ir::PointerT>().getPointeeType();
    auto [size, align] = type.getSizeAndAlign(structSizeMap);
    if (size < align)
      size = align;

    offsets.emplace_back(frameSize);
    frameSize += size;
  }

  if (frameSize > 0) {
    frameMem = llvm::safe_malloc(frameSize);
  }

  size_t i = 0;
  for (InstructionStorage *inst : *function->getAllocationBlock()) {
    inst::LocalVariable localVar = inst->getDefiningInst<inst::LocalVariable>();
    size_t offset = offsets[i++];
    auto localVarReg = createVRegister(localVar.getType(), structSizeMap);
    localVarReg->cast<VRegisterInt>()->setValue(
        localVars[localVar] = VMemory(static_cast<char *>(frameMem) + offset));
    registers[localVar] = localVarReg.get();
    stackRegisters.emplace_back(std::move(localVarReg));
  }

  function->walk([&](InstructionStorage *inst) -> WalkResult {
    for (ir::Value result : inst->getResults()) {
      auto reg = createVRegister(result.getType(), structSizeMap);
      registers[result] = reg.get();
      stackRegisters.emplace_back(std::move(reg));
    }
    return WalkResult::advance();
  });
}

std::unique_ptr<VRegister> StackFrame::constValue(inst::Constant c) const {
  auto type = c.getType();
  auto value = c.getValue();

  return llvm::TypeSwitch<ConstantAttr, std::unique_ptr<VRegister>>(value)
      .Case([&](ConstantIntAttr intAttr) {
        auto bitWidth = type.cast<ir::IntT>().getBitWidth();
        auto isSigned = type.cast<ir::IntT>().isSigned();
        if (bitWidth == 1)
          isSigned = false;

        llvm::APInt apInt(bitWidth, intAttr.getValue(), isSigned);
        llvm::APSInt apSInt(apInt, !isSigned);

        auto intR = std::make_unique<VRegisterInt>();
        intR->setValue(apSInt);
        return intR;
      })
      .Case([&](ConstantFloatAttr floatAttr) {
        auto floatR = std::make_unique<VRegisterFloat>();
        floatR->setValue(floatAttr.getValue());
        return floatR;
      })
      .Case([&](ConstantStringFloatAttr strAttr) {
        auto floatT = strAttr.getFloatType();
        llvm::APFloat apFloat(floatT.getBitWidth() == 32
                                  ? llvm::APFloat::IEEEsingle()
                                  : llvm::APFloat::IEEEdouble(),
                              strAttr.getValue());
        auto floatR = std::make_unique<VRegisterFloat>();
        floatR->setValue(apFloat);
        return floatR;
      })
      .Case([&](ConstantVariableAttr varAttr) {
        auto ptrR = std::make_unique<VRegisterInt>();
        auto globalVar = globalTable->getGlobal(varAttr.getName());
        assert(globalVar && "Global variable not found");
        ptrR->mv(globalVar);
        return ptrR;
      })
      .Case([&](ConstantUnitAttr unitAttr) {
        auto intR = std::make_unique<VRegisterInt>();
        intR->setValue(0);
        return intR;
      })
      .Case([&](ConstantUndefAttr undefAttr) -> std::unique_ptr<VRegister> {
        if (type.isa<ir::FloatT>()) {
          std::uint64_t undef;
          auto floatR = std::make_unique<VRegisterFloat>();
          floatR->setValue(undef);
          return floatR;
        } else if (type.isa<NameStruct, ArrayT>()) {
          auto [size, align] = type.getSizeAndAlign(
              interpreter->structSizeAnalysis->getStructSizeMap());
          if (size < align)
            size = align;
          return std::make_unique<VRegisterDynamic>(size);
        } else {
          std::uint64_t undef;
          auto intR = std::make_unique<VRegisterInt>();
          intR->setValue(undef);
          return intR;
        }
      })
      .Default([&](ConstantAttr) -> std::unique_ptr<VRegister> {
        llvm_unreachable("Unsupported constant type");
      });
}

void StackFrame::print(IRPrintContext &context, bool summary) const {
  context.getOS() << "function " << function->getName() << " called\n";
  if (callSite.isValid()) {
    auto &diag = function->getContext()->diag();
    auto &currOS = diag.getOS();
    auto exit = llvm::make_scope_exit([&]() { diag.setOS(&currOS); });

    diag.setOS(&context.getOS());
    function->getContext()->diag().report(callSite, llvm::SourceMgr::DK_Note,
                                          "Called from here");
  } else {
    context.getOS() << "First frame (entry function)" << '\n';
  }
  if (!summary) {
    context.getOS() << "Full stackFrame for function " << function->getName()
                    << ":n";
    for (const auto &[var, vmem] : localVars) {
      context.getOS() << '\n';
      auto rid = context.getId(var);

      context.getOS() << rid.toString() << ":" << var.getType() << " = ";
      registers.lookup(var)->print(context.getOS(), var.getType(),
                                   interpreter->structSizeAnalysis);
      context.getOS() << " -- inMemory(";
      vmem.print(context.getOS(),
                 var.getType().cast<PointerT>().getPointeeType(),
                 interpreter->structSizeAnalysis);
      context.getOS() << ")";
    }
    for (const auto &[val, vreg] : registers) {
      context.getOS() << '\n';
      auto rid = context.getId(val);
      context.getOS() << rid.toString() << ":" << val.getType() << " = ";
      vreg->print(context.getOS(), val.getType(),
                  interpreter->structSizeAnalysis);
    }
    context.getOS() << '\n';
  }
}

namespace impl {

#define ARITH_IMPL(Func, OpKind, operator)                                     \
  void Func(StackFrame *frame, ir::inst::Binary binary) {                      \
    assert(binary.getOpKind() == ir::inst::Binary::OpKind && "Only " #OpKind   \
                                                             " is supported"); \
    assert(binary.getLhs().getType().constCanonicalize() ==                    \
           binary.getRhs().getType().constCanonicalize());                     \
                                                                               \
    auto type = binary.getLhs().getType().constCanonicalize();                 \
                                                                               \
    InterpValue lhsReg = frame->getRegister(binary.getLhs());                  \
    InterpValue rhsReg = frame->getRegister(binary.getRhs());                  \
    InterpValue retReg = frame->getRegister(binary.getResult());               \
                                                                               \
    if (auto intT = type.dyn_cast<ir::IntT>()) {                               \
      auto isSigned = intT.isSigned();                                         \
      auto bitWidth = intT.getBitWidth();                                      \
      if (bitWidth == 1)                                                       \
        isSigned = false;                                                      \
                                                                               \
      VRegisterInt *lhsIntR = lhsReg->cast<VRegisterInt>();                    \
      VRegisterInt *rhsIntR = rhsReg->cast<VRegisterInt>();                    \
      VRegisterInt *retIntR = retReg->cast<VRegisterInt>();                    \
                                                                               \
      llvm::APSInt lhsInt = lhsIntR->getAsInteger(bitWidth, isSigned);         \
      llvm::APSInt rhsInt = rhsIntR->getAsInteger(bitWidth, isSigned);         \
      llvm::APSInt retInt = lhsInt operator rhsInt;                            \
      retIntR->setValue(retInt);                                               \
    } else if (auto floatT = type.dyn_cast<ir::FloatT>()) {                    \
      VRegisterFloat *lhsFloatR = lhsReg->cast<VRegisterFloat>();              \
      VRegisterFloat *rhsFloatR = rhsReg->cast<VRegisterFloat>();              \
      VRegisterFloat *retFloatR = retReg->cast<VRegisterFloat>();              \
                                                                               \
      auto bitwidth = floatT.getBitWidth();                                    \
      assert(bitwidth == 32 || bitwidth == 64);                                \
      auto lhsFloat = lhsFloatR->getAsFloat(bitwidth);                         \
      auto rhsFloat = rhsFloatR->getAsFloat(bitwidth);                         \
      auto retFloat = lhsFloat operator rhsFloat;                              \
      retFloatR->setValue(retFloat);                                           \
    } else {                                                                   \
      llvm_unreachable("Unsupported type for " #OpKind " instruction");        \
    }                                                                          \
  }

ARITH_IMPL(add, Add, +);
ARITH_IMPL(sub, Sub, -);
ARITH_IMPL(mul, Mul, *);
ARITH_IMPL(div, Div, /);

#undef ARITH_IMPL

#define INTEGER_OP_IMPL(Func, OpKind, operator)                                \
  void Func(StackFrame *frame, ir::inst::Binary binary) {                      \
    assert(binary.getOpKind() == ir::inst::Binary::OpKind && "Only " #OpKind   \
                                                             " is supported"); \
    assert(binary.getLhs().getType().constCanonicalize() ==                    \
           binary.getRhs().getType().constCanonicalize());                     \
                                                                               \
    auto type = binary.getLhs().getType().constCanonicalize();                 \
    assert(type.isa<ir::IntT>() && "Only integer types are supported");        \
                                                                               \
    auto intT = type.cast<ir::IntT>();                                         \
    auto isSigned = intT.isSigned();                                           \
    auto bitWidth = intT.getBitWidth();                                        \
    if (bitWidth == 1)                                                         \
      isSigned = false;                                                        \
                                                                               \
    InterpValue lhsReg = frame->getRegister(binary.getLhs());                  \
    InterpValue rhsReg = frame->getRegister(binary.getRhs());                  \
    InterpValue retReg = frame->getRegister(binary.getResult());               \
                                                                               \
    VRegisterInt *lhsIntR = lhsReg->cast<VRegisterInt>();                      \
    VRegisterInt *rhsIntR = rhsReg->cast<VRegisterInt>();                      \
    VRegisterInt *retIntR = retReg->cast<VRegisterInt>();                      \
                                                                               \
    llvm::APSInt lhsInt = lhsIntR->getAsInteger(bitWidth, isSigned);           \
    llvm::APSInt rhsInt = rhsIntR->getAsInteger(bitWidth, isSigned);           \
    llvm::APSInt retInt = lhsInt operator rhsInt;                              \
    retIntR->setValue(retInt);                                                 \
  }

INTEGER_OP_IMPL(mod, Mod, %);
INTEGER_OP_IMPL(bitAnd, BitAnd, &);
INTEGER_OP_IMPL(bitOr, BitOr, |);
INTEGER_OP_IMPL(bitXor, BitXor, ^);

#undef INTEGER_OP_IMPL

#define SHIFT_OP_IMPL(Func, OpKind, operator)                                  \
  void Func(StackFrame *frame, ir::inst::Binary binary) {                      \
    assert(binary.getOpKind() == ir::inst::Binary::OpKind && "Only " #OpKind   \
                                                             " is supported"); \
    assert(binary.getLhs().getType().constCanonicalize() ==                    \
           binary.getRhs().getType().constCanonicalize());                     \
                                                                               \
    auto type = binary.getLhs().getType().constCanonicalize();                 \
    assert(type.isa<ir::IntT>() && "Only integer types are supported");        \
                                                                               \
    auto intT = type.cast<ir::IntT>();                                         \
    auto isSigned = intT.isSigned();                                           \
    auto bitWidth = intT.getBitWidth();                                        \
    if (bitWidth == 1)                                                         \
      isSigned = false;                                                        \
                                                                               \
    InterpValue lhsReg = frame->getRegister(binary.getLhs());                  \
    InterpValue rhsReg = frame->getRegister(binary.getRhs());                  \
    InterpValue retReg = frame->getRegister(binary.getResult());               \
                                                                               \
    VRegisterInt *lhsIntR = lhsReg->cast<VRegisterInt>();                      \
    VRegisterInt *rhsIntR = rhsReg->cast<VRegisterInt>();                      \
    VRegisterInt *retIntR = retReg->cast<VRegisterInt>();                      \
                                                                               \
    llvm::APSInt lhsInt = lhsIntR->getAsInteger(bitWidth, isSigned);           \
    llvm::APSInt rhsInt = rhsIntR->getAsInteger(bitWidth, isSigned);           \
    llvm::APSInt retInt = lhsInt operator rhsInt.getZExtValue();               \
    retIntR->setValue(retInt);                                                 \
  }

SHIFT_OP_IMPL(shl, Shl, <<);
SHIFT_OP_IMPL(shr, Shr, >>);

#undef SHIFT_OP_IMPL

#define CMP_OP_IMPL(Func, OpKind, operator)                                    \
  void Func(StackFrame *frame, ir::inst::Binary binary) {                      \
    assert(binary.getOpKind() == ir::inst::Binary::OpKind && "Only " #OpKind   \
                                                             " is supported"); \
    assert(binary.getLhs().getType().constCanonicalize() ==                    \
           binary.getRhs().getType().constCanonicalize());                     \
                                                                               \
    auto type = binary.getLhs().getType().constCanonicalize();                 \
                                                                               \
    InterpValue lhsReg = frame->getRegister(binary.getLhs());                  \
    InterpValue rhsReg = frame->getRegister(binary.getRhs());                  \
    InterpValue retReg = frame->getRegister(binary.getResult());               \
                                                                               \
    VRegisterInt *retIntR = retReg->cast<VRegisterInt>();                      \
                                                                               \
    if (auto intT = type.dyn_cast<ir::IntT>()) {                               \
      auto isSigned = intT.isSigned();                                         \
      auto bitWidth = intT.getBitWidth();                                      \
      if (bitWidth == 1)                                                       \
        isSigned = false;                                                      \
                                                                               \
      VRegisterInt *lhsIntR = lhsReg->cast<VRegisterInt>();                    \
      VRegisterInt *rhsIntR = rhsReg->cast<VRegisterInt>();                    \
                                                                               \
      llvm::APSInt lhsInt = lhsIntR->getAsInteger(bitWidth, isSigned);         \
      llvm::APSInt rhsInt = rhsIntR->getAsInteger(bitWidth, isSigned);         \
      retIntR->setValue(lhsInt operator rhsInt ? 1 : 0);                       \
    } else if (auto floatT = type.dyn_cast<ir::FloatT>()) {                    \
      VRegisterFloat *lhsFloatR = lhsReg->cast<VRegisterFloat>();              \
      VRegisterFloat *rhsFloatR = rhsReg->cast<VRegisterFloat>();              \
                                                                               \
      auto bitwidth = floatT.getBitWidth();                                    \
      assert(bitwidth == 32 || bitwidth == 64);                                \
      auto lhsFloat = lhsFloatR->getAsFloat(bitwidth);                         \
      auto rhsFloat = rhsFloatR->getAsFloat(bitwidth);                         \
      retIntR->setValue(lhsFloat operator rhsFloat ? 1 : 0);                   \
    } else if (type.isa<ir::PointerT>()) {                                     \
      VRegisterInt *lhsIntR = lhsReg->cast<VRegisterInt>();                    \
      VRegisterInt *rhsIntR = rhsReg->cast<VRegisterInt>();                    \
      retIntR->setValue(lhsIntR->getValue() operator rhsIntR->getValue() ? 1   \
                                                                         : 0); \
    } else {                                                                   \
      llvm_unreachable("Unsupported type for " #OpKind " instruction");        \
    }                                                                          \
  }

CMP_OP_IMPL(eq, Eq, ==);
CMP_OP_IMPL(ne, Ne, !=);
CMP_OP_IMPL(lt, Lt, <);
CMP_OP_IMPL(le, Le, <=);
CMP_OP_IMPL(gt, Gt, >);
CMP_OP_IMPL(ge, Ge, >=);

#undef CMP_OP_IMPL

void unaryPlus(StackFrame *frame, ir::inst::Unary unary) {
  assert(unary.getOpKind() == ir::inst::Unary::Plus &&
         "Only Plus is supported");
  auto type = unary.getValue().getType().constCanonicalize();
  InterpValue valReg = frame->getRegister(unary.getValue());
  InterpValue retReg = frame->getRegister(unary.getResult());
  if (type.isa<ir::IntT>()) {
    VRegisterInt *valIntR = valReg->cast<VRegisterInt>();
    VRegisterInt *retIntR = retReg->cast<VRegisterInt>();
    retIntR->setValue(valIntR->getValue());
  } else if (type.isa<ir::FloatT>()) {
    VRegisterFloat *valFloatR = valReg->cast<VRegisterFloat>();
    VRegisterFloat *retFloatR = retReg->cast<VRegisterFloat>();
    retFloatR->setValue(valFloatR->getValue());
  } else {
    llvm_unreachable("Unsupported type for Plus instruction");
  }
}

void unaryMinus(StackFrame *frame, ir::inst::Unary unary) {
  assert(unary.getOpKind() == ir::inst::Unary::Minus &&
         "Only Minus is supported");
  auto type = unary.getValue().getType().constCanonicalize();
  InterpValue valReg = frame->getRegister(unary.getValue());
  InterpValue retReg = frame->getRegister(unary.getResult());
  if (type.isa<ir::IntT>()) {
    auto intT = type.cast<ir::IntT>();
    auto isSigned = intT.isSigned();
    auto bitWidth = intT.getBitWidth();
    if (bitWidth == 1)
      isSigned = false;

    VRegisterInt *valIntR = valReg->cast<VRegisterInt>();
    VRegisterInt *retIntR = retReg->cast<VRegisterInt>();

    llvm::APSInt valInt = valIntR->getAsInteger(bitWidth, isSigned);
    llvm::APSInt retInt = -valInt;
    retIntR->setValue(retInt);
  } else if (type.isa<ir::FloatT>()) {
    VRegisterFloat *valFloatR = valReg->cast<VRegisterFloat>();
    VRegisterFloat *retFloatR = retReg->cast<VRegisterFloat>();

    auto floatT = type.cast<ir::FloatT>();
    auto bitwidth = floatT.getBitWidth();
    assert(bitwidth == 32 || bitwidth == 64);
    auto valFloat = valFloatR->getAsFloat(bitwidth);
    auto retFloat = -valFloat;
    retFloatR->setValue(retFloat);
  } else {
    llvm_unreachable("Unsupported type for Minus instruction");
  }
}

void unaryNegate(StackFrame *frame, ir::inst::Unary unary) {
  assert(unary.getOpKind() == ir::inst::Unary::Negate &&
         "Only Negate is supported");
  auto type = unary.getValue().getType().constCanonicalize();
  assert(type.isa<ir::IntT>() && "Only integer types are supported");

  auto intT = type.cast<ir::IntT>();
  auto isSigned = intT.isSigned();
  auto bitWidth = intT.getBitWidth();
  if (bitWidth == 1)
    isSigned = false;

  InterpValue valReg = frame->getRegister(unary.getValue());
  InterpValue retReg = frame->getRegister(unary.getResult());

  VRegisterInt *valIntR = valReg->cast<VRegisterInt>();
  VRegisterInt *retIntR = retReg->cast<VRegisterInt>();

  llvm::APSInt valInt = valIntR->getAsInteger(bitWidth, isSigned);
  llvm::APSInt retInt = ~valInt;
  retIntR->setValue(retInt);
}

void gep(StackFrame *frame, ir::inst::Gep inst) {
  assert(inst.getBasePointer().getType().isa<ir::PointerT>() &&
         "Base pointer must be a pointer type");
  assert(inst.getOffset().getType().isa<ir::IntT>() &&
         "Offset must be an integer type");

  InterpValue baseReg = frame->getRegister(inst.getBasePointer());
  InterpValue offsetReg = frame->getRegister(inst.getOffset());
  InterpValue retReg = frame->getRegister(inst.getResult());

  VRegisterInt *baseIntR = baseReg->cast<VRegisterInt>();
  VRegisterInt *offsetIntR = offsetReg->cast<VRegisterInt>();
  VRegisterInt *retIntR = retReg->cast<VRegisterInt>();

  auto basePtr = baseIntR->getAsMemory();
  auto offset = offsetIntR->getValue();
  auto retPtr = basePtr.getElementPtr(offset);
  retIntR->setValue(retPtr);
}

void load(Interpreter *interpreter, inst::Load inst) {
  assert(inst.getPointer().getType().isa<ir::PointerT>() &&
         "Pointer must be a pointer type");

  assert(inst.getResult().getType() == inst.getPointer()
                                           .getType()
                                           .cast<ir::PointerT>()
                                           .getPointeeType()
                                           .constCanonicalize() &&
         "Result type must match the pointee type of the pointer");
  StackFrame *frame = interpreter->getCurrentFrame();
  InterpValue ptrReg = frame->getRegister(inst.getPointer());
  InterpValue retReg = frame->getRegister(inst.getResult());

  VRegisterInt *ptrIntR = ptrReg->cast<VRegisterInt>();
  VMemory ptr = ptrIntR->getAsMemory();

  ptr.loadInto(retReg.get(), inst.getResult().getType(),
               interpreter->getStructSizeAnalysis());
}

void store(Interpreter *interpreter, inst::Store inst) {
  assert(inst.getPointer().getType().isa<ir::PointerT>() &&
         "Pointer must be a pointer type");
  assert(inst.getValue().getType() == inst.getPointer()
                                          .getType()
                                          .cast<ir::PointerT>()
                                          .getPointeeType()
                                          .constCanonicalize() &&
         "Value type must match the pointee type of the pointer");
  StackFrame *frame = interpreter->getCurrentFrame();
  InterpValue srcReg = frame->getRegister(inst.getValue());
  InterpValue ptrReg = frame->getRegister(inst.getPointer());

  VRegisterInt *ptrIntR = ptrReg->cast<VRegisterInt>();
  VMemory ptr = ptrIntR->getAsMemory();

  ptr.storeFrom(srcReg.get(), inst.getValue().getType(),
                interpreter->getStructSizeAnalysis());
}

void typecast(StackFrame *frame, ir::inst::TypeCast typecast) {
  auto value = typecast.getValue();
  auto srcType = value.getType();
  auto destType = typecast.getResult().getType().constCanonicalize();

  InterpValue srcReg = frame->getRegister(value);
  InterpValue destReg = frame->getRegister(typecast.getResult());

  if (srcType == destType) {
    destReg->mv(srcReg.get());
    return;
  }

  if (srcType.isa<ir::PointerT>()) {
    if (destType.isa<ir::PointerT>()) {
      destReg->mv(srcReg.get());
      return;
    } else if (auto destIntT = destType.dyn_cast<IntT>()) {
      VRegisterInt *srcIntR = srcReg->cast<VRegisterInt>();
      llvm::APSInt srcInt =
          srcIntR->getAsInteger(destIntT.getBitWidth(), destIntT.isSigned());
      VRegisterInt *destIntR = destReg->cast<VRegisterInt>();
      destIntR->setValue(srcInt);
      return;
    }
  } else if (auto srcIntT = srcType.dyn_cast<IntT>()) {
    VRegisterInt *srcIntR = srcReg->cast<VRegisterInt>();
    if (auto destIntT = destType.dyn_cast<IntT>()) {
      llvm::APSInt srcInt =
          srcIntR->getAsInteger(destIntT.getBitWidth(), destIntT.isSigned());
      VRegisterInt *destIntR = destReg->cast<VRegisterInt>();
      destIntR->setValue(srcInt);
      return;
    } else if (destType.isa<ir::PointerT>()) {
      VRegisterInt *destIntR = destReg->cast<VRegisterInt>();
      destIntR->setValue(srcIntR->getValue());
      return;
    } else if (auto destFloatT = destType.dyn_cast<ir::FloatT>()) {
      auto srcInt =
          srcIntR->getAsInteger(srcIntT.getBitWidth(), srcIntT.isSigned());

      llvm::APFloat destFloat(destFloatT.getBitWidth() == 32
                                  ? llvm::APFloat::IEEEsingle()
                                  : llvm::APFloat::IEEEdouble());
      bool losesInfo = false;
      auto status = destFloat.convertFromAPInt(srcInt, srcInt.isSigned(),
                                               llvm::APFloat::rmTowardZero);
      assert(status == llvm::APFloat::opOK && "Conversion failed");
      // TODO: handle losesInfo?

      VRegisterFloat *destFloatR = destReg->cast<VRegisterFloat>();
      destFloatR->setValue(destFloat);
      return;
    }
  } else if (auto srcFloatT = srcType.dyn_cast<FloatT>()) {
    VRegisterFloat *srcFloatR = srcReg->cast<VRegisterFloat>();
    llvm::APFloat srcFloat = srcFloatR->getAsFloat(srcFloatT.getBitWidth());

    if (auto destIntT = destType.dyn_cast<IntT>()) {
      llvm::APSInt destInt(destIntT.getBitWidth(), !destIntT.isSigned());

      bool isExact = false;
      auto status = srcFloat.convertToInteger(
          destInt, llvm::APFloat::rmTowardZero, &isExact);
      assert(status == llvm::APFloat::opOK && "Conversion failed");

      VRegisterInt *destIntR = destReg->cast<VRegisterInt>();
      destIntR->setValue(destInt);
      return;
    } else if (auto destFloatT = destType.dyn_cast<FloatT>()) {
      bool losesInfo = false;
      auto status = srcFloat.convert(
          destFloatT.getBitWidth() == 32 ? llvm::APFloat::IEEEsingle()
                                         : llvm::APFloat::IEEEdouble(),
          llvm::APFloat::rmNearestTiesToEven, &losesInfo);
      assert(status == llvm::APFloat::opOK && "Conversion failed");
      auto destFloatR = destReg->cast<VRegisterFloat>();
      destFloatR->setValue(srcFloat);
      return;
    }
  }
}

void call(Interpreter *interpreter, inst::Call inst) {
  StackFrame *frame = interpreter->getCurrentFrame();
  InterpValue calleeReg = frame->getRegister(inst.getFunction());
  VRegisterInt *calleeIntR = calleeReg->cast<VRegisterInt>();
  auto calleeFunc = reinterpret_cast<ir::Function *>(calleeIntR->getValue());

  llvm::SmallVector<InterpValue> argValues;
  llvm::SmallVector<VRegister *> rawArgs;
  argValues.reserve(inst.getArguments().size());
  rawArgs.reserve(inst.getArguments().size());

  for (ir::Value argVal : inst.getArguments()) {
    InterpValue argReg = frame->getRegister(argVal);
    rawArgs.emplace_back(argReg.get());
    argValues.emplace_back(std::move(argReg));
  }

  llvm::SmallVector<std::unique_ptr<VRegister>> retValues =
      interpreter->call(calleeFunc->getName(), rawArgs, inst->getRange());

  for (size_t i = 0; i < inst->getResultSize(); ++i) {
    InterpValue retReg = frame->getRegister(inst.getResult(i));
    retReg->mv(retValues[i].get());
  }
}

void nop(ir::inst::Nop) {
  // Do nothing
}

void unreachable(Interpreter *interpreter, ir::inst::Unreachable unreachable) {
  interpreter->getModule()->getContext()->diag().report(
      unreachable->getRange(), llvm::SourceMgr::DK_Error, "((Unreachable))");

  interpreter->dumpAllStackFrames(llvm::errs());
  interpreter->setPC(nullptr);
}

static void jumpImpl(Interpreter *interpreter, StackFrame *frame,
                     ir::JumpArg *arg) {
  Block *nextBlock = arg->getBlock();

  auto iter = nextBlock->begin();
  for (size_t idx = 0; (*iter)->getDefiningInst<Phi>(); iter++, idx++) {
    Phi phi = (*iter)->getDefiningInst<Phi>();
    InterpValue incomingReg = frame->getRegister(arg->getArgs()[idx]);
    InterpValue phiReg = frame->getRegister(phi);
    phiReg->mv(incomingReg.get());
  }

  interpreter->setPC(*iter);
}

void jump(Interpreter *interpreter, inst::Jump jump) {
  StackFrame *frame = interpreter->getCurrentFrame();
  JumpArg *arg = jump.getJumpArg();
  jumpImpl(interpreter, frame, arg);
}

void branch(Interpreter *interpreter, inst::Branch branch) {
  assert(branch.getCondition().getType().isa<ir::IntT>() &&
         "Condition must be an integer type");
  StackFrame *frame = interpreter->getCurrentFrame();
  InterpValue condReg = frame->getRegister(branch.getCondition());
  VRegisterInt *condIntR = condReg->cast<VRegisterInt>();

  bool isTrue = condIntR->getValue() != 0;

  JumpArg *nextArg = isTrue ? branch.getIfArg() : branch.getElseArg();
  jumpImpl(interpreter, frame, nextArg);
}

void switchExit(Interpreter *interpreter, inst::Switch switchExit) {
  assert(switchExit.getValue().getType().isa<ir::IntT>() &&
         "Condition must be an integer type");
  StackFrame *frame = interpreter->getCurrentFrame();
  InterpValue condReg = frame->getRegister(switchExit.getValue());
  VRegisterInt *condIntR = condReg->cast<VRegisterInt>();

  int64_t condValue = condIntR->getValue();
  JumpArg *nextArg = switchExit.getDefaultCase();

  for (auto idx = 0u; idx < switchExit.getCaseSize(); ++idx) {
    assert(switchExit.getCaseValue(idx).getType().isa<ir::IntT>() &&
           "Case value must be an integer type");
    InterpValue caseReg = frame->getRegister(switchExit.getCaseValue(idx));
    VRegisterInt *caseIntR = caseReg->cast<VRegisterInt>();
    if (caseIntR->getValue() == condValue) {
      nextArg = switchExit.getCaseJumpArg(idx);
      break;
    }
  }

  jumpImpl(interpreter, frame, nextArg);
}

void retExit(Interpreter *interpreter, inst::Return ret) {
  StackFrame *frame = interpreter->getCurrentFrame();
  auto &rets = frame->getReturnValues();
  assert(rets.empty() && "Return values should be empty");
  rets.reserve(ret.getValues().size());
  for (ir::Value retVal : ret.getValues()) {
    InterpValue retReg = frame->getRegister(retVal);
    rets.emplace_back(retReg->clone());
  }
}

} // namespace impl

static void initGlobalMemImpl(Type type, void *mem, Attribute init,
                              const StructSizeAnalysis *sizeMap,
                              size_t &filledSize) {
  auto isZero = !init;
  if (auto array = init.dyn_cast_or_null<ArrayAttr>()) {
    if (array.getValues().empty())
      isZero = true;
  }

  if (isZero) {
    auto [size, align] = type.getSizeAndAlign(sizeMap->getStructSizeMap());
    if (size < align)
      size = align;

    memset(mem, 0, size);
    filledSize += size;
    return;
  }

  auto [size, align] = type.getSizeAndAlign(sizeMap->getStructSizeMap());
  llvm::TypeSwitch<Type, void>(type)
      .Case([&](IntT intT) {
        assert(init.isa<ConstantIntAttr>() && "Initializer must be an integer");
        auto intInit = init.cast<ConstantIntAttr>();
        auto bitWidth = intInit.getBitWidth();
        auto value = intInit.getValue();
        if (bitWidth <= 8) {
          assert(bitWidth == 1 || bitWidth == 8);
          *static_cast<std::int8_t *>(mem) = static_cast<std::int8_t>(value);
        } else if (bitWidth == 16) {
          *static_cast<std::int16_t *>(mem) = static_cast<std::int16_t>(value);
        } else if (bitWidth == 32) {
          *static_cast<std::int32_t *>(mem) = static_cast<std::int32_t>(value);
        } else if (bitWidth == 64) {
          *static_cast<std::int64_t *>(mem) = static_cast<std::int64_t>(value);
        } else {
          llvm_unreachable("Unsupported integer bit width");
        }
        filledSize += size;
      })
      .Case([&](FloatT floatT) {
        assert(init.isa<ConstantFloatAttr>() && "Initializer must be a float");
        auto floatInit = init.cast<ConstantFloatAttr>();
        auto value = floatInit.getValue();
        auto bitValue = value.bitcastToAPInt().getZExtValue();
        if (&value.getSemantics() == &llvm::APFloat::IEEEsingle()) {
          *static_cast<std::uint32_t *>(mem) =
              static_cast<std::uint32_t>(bitValue);
        } else if (&value.getSemantics() == &llvm::APFloat::IEEEdouble()) {
          *static_cast<std::uint64_t *>(mem) =
              static_cast<std::uint64_t>(bitValue);
        } else {
          llvm_unreachable("Unsupported float bit width");
        }
        filledSize += size;
      })
      .Case([&](ArrayT arrayT) {
        auto arrayAttr = init.cast<ArrayAttr>();
        auto elemType = arrayT.getElementType();

        size_t arraySize = 0;
        auto [elemSize, elemAlign] =
            elemType.getSizeAndAlign(sizeMap->getStructSizeMap());
        if (elemSize < elemAlign)
          elemSize = elemAlign;

        for (const auto &[idx, value] :
             llvm::enumerate(arrayAttr.getValues())) {
          assert(arraySize == idx * elemSize);
          initGlobalMemImpl(elemType, static_cast<char *>(mem) + arraySize,
                            value, sizeMap, arraySize);
        }

        auto [arrayMemSize, arrayMemAlign] =
            arrayT.getSizeAndAlign(sizeMap->getStructSizeMap());
        if (arraySize < arrayMemSize) {
          memset(static_cast<char *>(mem) + arraySize, 0,
                 arrayMemSize - arraySize);
          arraySize = arrayMemSize;
        }
        filledSize += arraySize;
      })
      .Case([&](NameStruct structT) {
        auto arrayAttr = init.cast<ArrayAttr>();
        auto structName = structT.getName();
        const auto &[size, align, offsets] =
            sizeMap->getStructSizeMap().at(structName);

        const auto &fields = sizeMap->getStructFieldsMap().at(structName);

        size_t structSize = 0;
        for (const auto &[idx, field, offset] :
             llvm::enumerate(fields, offsets)) {
          if (structSize < offset) {
            memset(static_cast<char *>(mem) + structSize, 0,
                   offset - structSize);
            structSize = offset;
          }

          initGlobalMemImpl(field, static_cast<char *>(mem) + structSize,
                            idx < arrayAttr.getValues().size()
                                ? arrayAttr.getValues()[idx]
                                : nullptr,
                            sizeMap, structSize);
        }

        if (structSize < size) {
          memset(static_cast<char *>(mem) + structSize, 0, size - structSize);
          structSize = size;
        }

        filledSize += structSize;
      })
      .Default([&](Type) {
        llvm_unreachable("Unsupported type for global initializer");
      });
}

static void initGlobalMem(Type type, void *mem, Attribute attr,
                          const StructSizeAnalysis *sizeMap) {
  size_t filledSize = 0;
  initGlobalMemImpl(type, mem, attr, sizeMap, filledSize);
  auto [size, align] = type.getSizeAndAlign(sizeMap->getStructSizeMap());
  if (size < align)
    size = align;
  assert(filledSize == size && "Initializer size does not match the type size");
  (void)filledSize;
}

void Interpreter::initGlobal() {
  const auto &structSizeMap = structSizeAnalysis->getStructSizeMap();

  for (auto globalVar : *module->getIR()->getGlobalBlock()) {
    auto gv = globalVar->getDefiningInst<inst::GlobalVariableDefinition>();
    assert(gv && "Global block can only contain global variable definitions");

    auto type = gv.getType();
    auto pointerT = PointerT::get(type.getContext(), type);
    auto ptrReg = createVRegister(pointerT, structSizeMap);
    auto name = gv.getName();

    auto [size, align] = type.getSizeAndAlign(structSizeMap);
    void *mem = globalTable.getAllocator().Allocate(size, align);

    if (gv.hasInitializer()) {
      gv.interpretInitializer();
      assert(!gv->getContext()->diag().hasError());
    }
    initGlobalMem(type, mem, gv.getInitializer(), structSizeAnalysis);

    auto newPtrReg = std::make_unique<VRegisterInt>();
    newPtrReg->setValue(VMemory(mem));

    globalTable.addGlobal(name, std::move(newPtrReg));
  }

  for (auto func : *module->getIR()) {
    auto functionT = func->getFunctionType();
    auto newPtrReg = createVRegister(functionT, structSizeMap);
    newPtrReg->cast<VRegisterInt>()->setValue(
        reinterpret_cast<std::uint64_t>(func));
    auto name = func->getName();
    globalTable.addGlobal(name, std::move(newPtrReg));
  }
}

static void executeInstruction(Interpreter *interpreter, StackFrame *frame,
                               InstructionStorage *inst) {
  llvm::TypeSwitch<InstructionStorage *>(inst)
      .Case([&](inst::Binary binary) {
        switch (binary.getOpKind()) {
        case inst::Binary::Add:
          return impl::add(frame, binary);
        case inst::Binary::Sub:
          return impl::sub(frame, binary);
        case inst::Binary::Mul:
          return impl::mul(frame, binary);
        case inst::Binary::Div:
          return impl::div(frame, binary);
        case inst::Binary::Mod:
          return impl::mod(frame, binary);
        case inst::Binary::BitAnd:
          return impl::bitAnd(frame, binary);
        case inst::Binary::BitOr:
          return impl::bitOr(frame, binary);
        case inst::Binary::BitXor:
          return impl::bitXor(frame, binary);
        case inst::Binary::Shl:
          return impl::shl(frame, binary);
        case inst::Binary::Shr:
          return impl::shr(frame, binary);
        case inst::Binary::Eq:
          return impl::eq(frame, binary);
        case inst::Binary::Ne:
          return impl::ne(frame, binary);
        case inst::Binary::Lt:
          return impl::lt(frame, binary);
        case inst::Binary::Le:
          return impl::le(frame, binary);
        case inst::Binary::Gt:
          return impl::gt(frame, binary);
        case inst::Binary::Ge:
          return impl::ge(frame, binary);
        }
      })
      .Case([&](inst::Unary unary) {
        switch (unary.getOpKind()) {
        case inst::Unary::Plus:
          return impl::unaryPlus(frame, unary);
        case inst::Unary::Minus:
          return impl::unaryMinus(frame, unary);
        case inst::Unary::Negate:
          return impl::unaryNegate(frame, unary);
        }
      })
      .Case([&](inst::Gep gep) { return impl::gep(frame, gep); })
      .Case([&](inst::Load load) { return impl::load(interpreter, load); })
      .Case([&](inst::Store store) { return impl::store(interpreter, store); })
      .Case([&](inst::Call call) { return impl::call(interpreter, call); })
      .Case([&](inst::TypeCast typecast) {
        return impl::typecast(frame, typecast);
      })
      .Case([&](inst::Nop nop) { return impl::nop(nop); })
      .Case([&](inst::Unreachable unreachable) {
        return impl::unreachable(interpreter, unreachable);
      })
      .Case([&](inst::Jump jump) { return impl::jump(interpreter, jump); })
      .Case([&](inst::Branch branch) {
        return impl::branch(interpreter, branch);
      })
      .Case([&](inst::Switch switchExit) {
        return impl::switchExit(interpreter, switchExit);
      })
      .Case([&](inst::Return ret) { return impl::retExit(interpreter, ret); })
      .Default([&](InstructionStorage *inst) {
        llvm::errs() << "Unsupported instruction: " << inst->getInstName()
                     << '\n';
        llvm_unreachable("Unsupported instruction in interpreter");
      });
}

llvm::SmallVector<std::unique_ptr<VRegister>>
Interpreter::call(llvm::StringRef name, llvm::ArrayRef<VRegister *> args,
                  llvm::SMRange callSite) {
  auto func = module->getIR()->getFunction(name);
  assert(func && "Function not found");
  IRContext *irContext = func->getContext();

  llvm::raw_ostream &os = irContext->diag().getOS();
  if (!func->hasDefinition()) {
    os << "Function " << name << " has no definition\n";
    dumpAllStackFrames(os);
    llvm::report_fatal_error("Cannot call function without definition");
  }

  CallStack callStackDecl(this, func, callSite);
  if (callStack.size() > stackOverflowLimit) {
    os << "Stack overflow when calling function " << name << "\n";
    dumpShortenedStackFrames(os);
    setPC(nullptr);
    return {};
  }

  StackFrame *frame = getCurrentFrame();
  auto iter = func->getEntryBlock()->begin();
  size_t idx;
  for (idx = 0u; idx < args.size(); ++idx, ++iter) {
    auto *inst = *iter;
    assert(inst->getDefiningInst<Phi>() ||
           inst->getDefiningInst<inst::FunctionArgument>());
    auto argReg = frame->getRegister(inst->getResult(0));
    argReg->mv(args[idx]);
  }
  assert(idx == args.size() && "Argument size mismatch");

  PCGuard guard(this);
  setPC(*iter);

  while (getPC() && !irContext->diag().hasError()) {
    auto *currentInst = getPC();
    auto *nextInst = currentInst->getNextInBlock();
    setPC(nextInst);

    executeInstruction(this, frame, currentInst);
  }

  if (irContext->diag().hasError()) {
    std::unique_ptr<VRegisterInt> retR = std::make_unique<VRegisterInt>();
    retR->setValue(-1);
    frame->getReturnValues().clear();
    frame->getReturnValues().emplace_back(std::move(retR));
  }

  return std::move(frame->getReturnValues());
}

int Interpreter::callMain(llvm::ArrayRef<llvm::StringRef> args) {
  Function *mainFunc = module->getIR()->getFunction("main");
  if (!mainFunc) {
    module->getContext()->diag().getOS() << "Function 'main' not found in IR\n";
    return -1;
  }

  if (!mainFunc->hasDefinition()) {
    module->getContext()->diag().getOS()
        << "Function 'main' has no definition\n";
    return -1;
  }

  auto mainFuncT = mainFunc->getFunctionType().cast<FunctionT>();
  auto parmTypes = mainFuncT.getArgTypes();
  auto retTypes = mainFuncT.getReturnTypes();

  if (parmTypes.size() > 2 || retTypes.size() > 1 ||
      (parmTypes.size() == 2 &&
       (!parmTypes[0].isa<IntT>() || !parmTypes[1].isa<PointerT>())) ||
      (parmTypes.size() == 1 && !parmTypes[0].isa<IntT>()) ||
      (retTypes.size() == 1 && !retTypes[0].isa<IntT, UnitT>())) {
    module->getContext()->diag().getOS()
        << "Function 'main' has invalid signature\n";
    return -1;
  }

  std::unique_ptr<VRegisterInt> argcR, argvR;
  llvm::SmallVector<VRegister *, 2> mainArgs;
  if (parmTypes.size() >= 1) {
    argcR = std::make_unique<VRegisterInt>();
    argcR->setValue(args.size());
    mainArgs.emplace_back(argcR.get());
  }

  llvm::SmallVector<char> argBuffer;
  llvm::SmallVector<char *> argPtrs;

  if (parmTypes.size() == 2) {
    argvR = std::make_unique<VRegisterInt>();

    for (llvm::StringRef arg : args) {
      char *argPtr = argBuffer.data() + argBuffer.size();
      argBuffer.append(arg.begin(), arg.end());
      argBuffer.push_back('\0');
      argPtrs.push_back(argPtr);
    }

    char **argPtr = argPtrs.data();
    argvR->setValue(VMemory(argPtr));
  }

  printGlobalTable(llvm::errs());
  auto rets = call("main", mainArgs, llvm::SMRange());
  assert(rets.size() <= 1);
  if (rets.empty())
    return 0;

  VRegister *retR = rets[0].get();
  if (retTypes[0].isa<UnitT>())
    return 0;

  return retR->cast<VRegisterInt>()->getValue();
}

void Interpreter::printGlobalTable(llvm::raw_ostream &os) const {
  os << "Global Table:\n";
  for (auto *inst : *module->getIR()->getGlobalBlock()) {
    auto gv = inst->getDefiningInst<inst::GlobalVariableDefinition>();
    assert(gv && "Global block can only contain global variable definitions");
    auto name = gv.getName();
    os << "  @" << name << " = ";
    auto type = gv.getType();
    auto pointerT = PointerT::get(type.getContext(), type);
    auto reg = globalTable.getGlobal(name);
    auto vmem = reg->cast<VRegisterInt>()->getAsMemory();
    reg->print(os, pointerT, structSizeAnalysis);
    os << " -- inMemory(";
    vmem.print(os, type, structSizeAnalysis);
    os << ")\n";
  }
}

void Interpreter::dumpAllStackFrames(llvm::raw_ostream &os) const {
  IRPrintContext context(os);
  printGlobalTable(os);
  for (ir::Function *func : *module->getIR()) {
    if (!func->hasDefinition())
      continue;
    func->registerAllInstInfo(context);
  }

  for (const auto &stackFrame : llvm::reverse(callStack))
    stackFrame->print(context);
}

void Interpreter::dumpShortenedStackFrames(llvm::raw_ostream &os) const {
  IRPrintContext context(os);
  for (const auto &stackFrame : llvm::reverse(callStack))
    stackFrame->print(context, true);
}

} // namespace kecc::ir
