#include "ir/Context.h"
#include "ir/IRTypes.h"
#include "ir/Types.h"

namespace kecc::ir {

template <typename T> void registerType(IRContext *context) {
  utils::TypeId typeId = utils::getId<T>;
  auto *abstType = new (context->getTypeStorage()->allocate<AbstractType>())
      AbstractType(AbstractType::build<T>(context));
  context->getTypeStorage()->registerAbstractType(typeId, abstType);
}

IRContext::IRContext() : typeStorage(std::make_unique<TypeStorage>()) {
  // Register built-in types
  registerType<IntT>(this);
  registerType<FloatT>(this);
  registerType<NameStruct>(this);
  registerType<UnitT>(this);
  registerType<FunctionT>(this);
  registerType<PointerT>(this);
}

} // namespace kecc::ir
