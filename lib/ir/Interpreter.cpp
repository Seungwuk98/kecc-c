#include "kecc/ir/Interpreter.h"
#include "kecc/ir/IRInstructions.h"
#include "kecc/ir/IRTypes.h"
#include "kecc/ir/Instruction.h"
#include "kecc/ir/TypeAttributeSupport.h"
#include "kecc/ir/WalkSupport.h"
#include "llvm/Support/MemAlloc.h"

namespace kecc::ir {

StackFrame::~StackFrame() {
  if (frameMem)
    free(frameMem);
}

static std::unique_ptr<VRegister>
createVRegister(ir::Type type, const StructSizeMap &sizeMap) {
  if (type.isa<ir::FloatT>()) {
    return std::make_unique<VRegisterFloat>();
  } else if (auto structT = type.dyn_cast<NameStruct>()) {
    auto [size, align] = structT.getSizeAndAlign(sizeMap);
    if (size < align)
      size = align;
    return std::make_unique<VRegisterStruct>(size);
  } else {
    return std::make_unique<VRegisterInt>();
  }
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
    localVars[localVar] = VMemory(static_cast<char *>(frameMem) + offset);

    auto localVarReg = createVRegister(localVar.getType(), structSizeMap);
    registers[localVar] = std::move(localVarReg);
  }

  function->walk([&](InstructionStorage *inst) -> WalkResult {
    for (ir::Value result : inst->getResults()) {
      auto reg = createVRegister(result.getType(), structSizeMap);
      registers[result] = std::move(reg);
    }
    return WalkResult::advance();
  });
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
    VRegister *lhsReg = frame->getRegister(binary.getLhs());                   \
    VRegister *rhsReg = frame->getRegister(binary.getRhs());                   \
    VRegister *retReg = frame->getRegister(binary.getResult());                \
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

void mod(StackFrame *frame, ir::inst::Binary binary) {
  assert(binary.getOpKind() == ir::inst::Binary::Mod &&
         "Only Mod is supported");
  assert(binary.getLhs().getType().constCanonicalize() ==
         binary.getRhs().getType().constCanonicalize());

  auto type = binary.getLhs().getType().constCanonicalize();
  assert(type.isa<ir::IntT>() && "Mod is only supported for integer types");

  auto intT = type.cast<ir::IntT>();
  auto isSigned = intT.isSigned();
  auto bitWidth = intT.getBitWidth();
  if (bitWidth == 1)
    isSigned = false;

  VRegister *lhsReg = frame->getRegister(binary.getLhs());
  VRegister *rhsReg = frame->getRegister(binary.getRhs());
  VRegister *retReg = frame->getRegister(binary.getResult());

  VRegisterInt *lhsIntR = lhsReg->cast<VRegisterInt>();
  VRegisterInt *rhsIntR = rhsReg->cast<VRegisterInt>();
  VRegisterInt *retIntR = retReg->cast<VRegisterInt>();

  llvm::APSInt lhsInt = lhsIntR->getAsInteger(bitWidth, isSigned);
  llvm::APSInt rhsInt = rhsIntR->getAsInteger(bitWidth, isSigned);
  llvm::APSInt retInt = lhsInt % rhsInt;
  retIntR->setValue(retInt);
}

} // namespace impl

} // namespace kecc::ir
