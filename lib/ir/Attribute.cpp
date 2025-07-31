#include "kecc/ir/Attribute.h"

namespace kecc::ir {

TypeID Attribute::getId() const { return impl->getId(); }

IRContext *Attribute::getContext() const { return impl->getContext(); }

void Attribute::walkSubElements(
    const llvm::function_ref<void(Type)> typeWalkFn,
    const llvm::function_ref<void(Attribute)> attrWalkFn) const {
  impl->getAbstractAttribute()->getWalkSubElementsFn()(*this, typeWalkFn,
                                                       attrWalkFn);
}

Attribute
Attribute::replaceSubElements(llvm::ArrayRef<Type> typeReplaced,
                              llvm::ArrayRef<Attribute> attrReplaced) const {
  return impl->getAbstractAttribute()->getReplaceSubElementsFn()(
      *this, typeReplaced, attrReplaced);
}

} // namespace kecc::ir
