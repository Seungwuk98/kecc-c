#include "kecc/translate/FunctionStack.h"

namespace kecc {

size_t StackPoint::fromBottom() const {
  assert(stack != nullptr);
  return stack->fromBottom(*this);
}

size_t FunctionStack::fromBottom(const StackPoint &point) const {
  size_t offset = 0;

  if (point.area == StackPoint::Area::CallArguments) {
    offset += point.offset;
    return offset;
  } else
    offset += callArgumentsSize;

  if (point.area == StackPoint::Area::CallerSavedRegisters) {
    offset += point.offset;
    return offset;
  } else
    offset += callerSavedRegistersSize;

  if (point.area == StackPoint::Area::CalleeSavedRegisters) {
    offset += point.offset;
    return offset;
  } else
    offset += calleeSavedRegistersSize;

  if (point.area == StackPoint::Area::SpilledRegisters) {
    offset += point.offset;
    return offset;
  } else
    offset += spilledRegistersSize;

  assert(point.offset != 0);
  return offset;
}

} // namespace kecc
