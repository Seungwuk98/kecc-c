#include "kecc/ir/Attribute.h"

namespace kecc::ir {

TypeID Attribute::getId() const { return impl->getId(); }

IRContext *Attribute::getContext() const { return impl->getContext(); }

} // namespace kecc::ir
