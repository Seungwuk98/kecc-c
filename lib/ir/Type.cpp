#include "kecc/ir/Type.h"
#include "kecc/ir/Context.h"
#include "kecc/ir/IRTypes.h"

namespace kecc::ir {

std::string Type::toString() const {
  std::string result;
  llvm::raw_string_ostream os(result);
  print(os);
  return result;
}

void Type::print(llvm::raw_ostream &os) const {
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

} // namespace kecc::ir
