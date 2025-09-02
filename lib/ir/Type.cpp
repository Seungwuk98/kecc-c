#include "kecc/ir/Type.h"
#include "kecc/ir/Context.h"
#include "kecc/ir/IRTypes.h"
#include "kecc/ir/TypeWalk.h"
#include "mlir/Support/LLVM.h"

namespace kecc::ir {

std::string Type::toString() const {
  std::string result;
  llvm::raw_string_ostream os(result);
  print(os);
  return result;
}

void Type::print(llvm::raw_ostream &os) const {
  if (!*this) {
    os << "<Null>";
    return;
  }
  getImpl()->getPrintFn()(*this, os);
}

TypeID Type::getId() const { return impl->getId(); }

Type Type::constCanonicalize() const {
  if (auto constQual = dyn_cast<ConstQualifier>())
    return constQual.getType();

  if (auto pointer = dyn_cast<PointerT>())
    return PointerT::get(getContext(), pointer.getPointeeType(), false);
  return *this;
}

IRContext *Type::getContext() const { return getImpl()->getContext(); }

void Type::walkSubElements(
    const llvm::function_ref<void(Type)> typeWalkFn,
    const llvm::function_ref<void(Attribute)> attrWalkFn) const {
  getImpl()->getWalkSubElementsFn()(*this, typeWalkFn, attrWalkFn);
}

Type Type::replaceSubElements(llvm::ArrayRef<Type> typeReplaced,
                              llvm::ArrayRef<Attribute> attrReplaced) const {
  return getImpl()->getReplaceSubElementsFn()(*this, typeReplaced,
                                              attrReplaced);
}

std::pair<size_t, size_t>
Type::getSizeAndAlign(const StructSizeMap &sizeMap) const {
  return getImpl()->getSizeAndAlignFn()(*this, sizeMap);
}

bool Type::isSignedInt() const {
  if (auto intType = dyn_cast<IntT>())
    return intType.isSigned();
  return false;
}

} // namespace kecc::ir
