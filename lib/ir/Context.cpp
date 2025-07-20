#include "kecc/ir/Context.h"
#include "kecc/ir/Attribute.h"
#include "kecc/ir/IRAttributes.h"
#include "kecc/ir/IRInstructions.h"
#include "kecc/ir/IRTypes.h"
#include "kecc/ir/Type.h"
#include "kecc/utils/Diag.h"
#include "llvm/Support/raw_ostream.h"

namespace kecc::ir {

template <typename T> void registerType(IRContext *context) {
  TypeID typeId = TypeID::get<T>();
  auto *abstType = new (context->getTypeStorage()->allocate<AbstractType>())
      AbstractType(AbstractType::build<T>(context));
  context->getTypeStorage()->registerType<T>(abstType);
}

template <typename T> void registerAttr(IRContext *context) {
  TypeID typeId = TypeID::get<T>();
  auto *abstAttr =
      new (context->getTypeStorage()->allocate<AbstractAttribute>())
          AbstractAttribute(AbstractAttribute::build<T>(context));
  context->getTypeStorage()->registerAttr<T>(abstAttr);
}

template <typename T> void registerInst(IRContext *context) {
  TypeID typeId = TypeID::get<T>();
  auto *abstInst =
      new (context->getTypeStorage()->allocate<AbstractInstruction>())
          AbstractInstruction(AbstractInstruction::build<T>(context));
  context->getTypeStorage()->registerInst<T>(abstInst);
}

IRContext::IRContext()
    : typeStorage(std::make_unique<TypeStorage>(this)), diagEngine(srcMgr) {
  // Register built-in types
  registerType<IntT>(this);
  registerType<FloatT>(this);
  registerType<NameStruct>(this);
  registerType<UnitT>(this);
  registerType<FunctionT>(this);
  registerType<PointerT>(this);
  registerType<ConstQualifier>(this);
  registerType<TupleT>(this);
  registerType<ArrayT>(this);

  // Register built-in attributes
  registerAttr<StringAttr>(this);
  registerAttr<ConstantIntAttr>(this);
  registerAttr<ConstantFloatAttr>(this);
  registerAttr<ConstantStringFloatAttr>(this);
  registerAttr<ConstantUndefAttr>(this);
  registerAttr<ConstantUnitAttr>(this);
  registerAttr<ConstantVariableAttr>(this);
  registerAttr<ArrayAttr>(this);
  registerAttr<TypeAttr>(this);
  registerAttr<EnumAttr>(this);
  registerAttr<ASTInitializerList>(this);
  registerAttr<ASTGroupOp>(this);
  registerAttr<ASTUnaryOp>(this);
  registerAttr<ASTInteger>(this);
  registerAttr<ASTFloat>(this);

  // Register built-in instructions
  registerInst<Phi>(this);
  registerInst<inst::Nop>(this);
  registerInst<inst::Load>(this);
  registerInst<inst::Store>(this);
  registerInst<inst::Call>(this);
  registerInst<inst::TypeCast>(this);
  registerInst<inst::Gep>(this);
  registerInst<inst::Binary>(this);
  registerInst<inst::Unary>(this);
  registerInst<inst::Jump>(this);
  registerInst<inst::Branch>(this);
  registerInst<inst::Switch>(this);
  registerInst<inst::Return>(this);
  registerInst<inst::Unreachable>(this);
  registerInst<inst::Constant>(this);
  registerInst<inst::StructDefinition>(this);
  registerInst<inst::GlobalVariableDefinition>(this);
  registerInst<inst::LocalVariable>(this);
  registerInst<inst::Unresolved>(this);
}

IRContext::~IRContext() {
  for (void *allocation : allocations) {
    free(allocation);
  }
}

std::string RegisterId::toString() const {
  std::string result;
  llvm::raw_string_ostream ss(result);

  ss << '%';
  switch (getKind()) {
  case Arg:
    ss << 'b' << getBlockId() << ':' << 'p';
    break;
  case Temp:
    ss << 'b' << getBlockId() << ':' << 'i';
    break;
  case Alloc:
    ss << 'l';
    break;
  case Unresolved:
    ss << 'U';
    break;
  }

  ss << getRegId();
  return result;
}

void *IRContext::allocate(size_t size) {
  void *ptr = llvm::safe_malloc(size);
  allocations.emplace_back(ptr);
  return ptr;
}

RegisterId RegisterId::arg(llvm::SMRange range, int blockId,
                           std::size_t regId) {
  return RegisterId(range, blockId, Kind::Arg, regId);
}

RegisterId RegisterId::temp(llvm::SMRange range, int blockId,
                            std::size_t regId) {
  return RegisterId(range, blockId, Kind::Temp, regId);
}

RegisterId RegisterId::alloc(llvm::SMRange range, std::size_t regId) {
  return RegisterId(range, RegisterId::AllocBlockId, Kind::Alloc, regId);
}

RegisterId RegisterId::unresolved(llvm::SMRange range, std::size_t regId) {
  return RegisterId(range, 0, Kind::Unresolved, regId);
}

} // namespace kecc::ir
