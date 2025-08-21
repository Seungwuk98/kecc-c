#ifndef KECC_TRANSLATE_FUNCTION_STACK_H
#define KECC_TRANSLATE_FUNCTION_STACK_H

#include <cassert>
#include <cstddef>
namespace kecc {

class FunctionStack;

struct StackPoint {
  StackPoint() = default;
  StackPoint(const StackPoint &) = default;
  StackPoint &operator=(const StackPoint &) = default;

  enum Area {
    ReturnAddress,        // 0 or 8 bytes
    SpilledRegisters,     // spilled registers
    CalleeSavedRegisters, // callee saved registers
    CallerSavedRegisters, // caller saved registers
    CallArguments         // call arguments
  };

  StackPoint(FunctionStack *stack, Area area, size_t offset)
      : stack(stack), area(area), offset(offset) {}

  size_t fromBottom() const;

  FunctionStack *stack;
  Area area;
  size_t offset; // offset from the bottom of the stack
};

/// FunctionStack is a class that represents the stack of a function.
///
///--------------------------------------------------------------------------///
/// Top
///--------------------------------------------------------------------------///
/// return address (0 or 8 bytes)
///--------------------------------------------------------------------------///
/// spilled registers
///--------------------------------------------------------------------------///
/// callee saved registers
///--------------------------------------------------------------------------///
/// caller saved registers
///--------------------------------------------------------------------------///
/// call arguments
///--------------------------------------------------------------------------///
class FunctionStack {
public:
  FunctionStack() = default;

  size_t getReturnAddressSize() const { return returnAddressSize; }
  size_t getSpilledRegistersSize() const { return spilledRegistersSize; }
  size_t getCalleeSavedRegistersSize() const {
    return calleeSavedRegistersSize;
  }
  size_t getCallerSavedRegistersSize() const {
    return callerSavedRegistersSize;
  }
  size_t getCallArgumentsSize() const { return callArgumentsSize; }

  size_t getTotalSize() const {
    return returnAddressSize + spilledRegistersSize + calleeSavedRegistersSize +
           callerSavedRegistersSize + callArgumentsSize;
  }

  void setReturnAddressSize(size_t size) { returnAddressSize = size; }
  void setSpilledRegistersSize(size_t size) { spilledRegistersSize = size; }
  void setCalleeSavedRegistersSize(size_t size) {
    calleeSavedRegistersSize = size;
  }
  void setCallerSavedRegistersSize(size_t size) {
    callerSavedRegistersSize = size;
  }
  void setCallArgumentsSize(size_t size) { callArgumentsSize = size; }

  size_t fromBottom(const StackPoint &point) const;

  StackPoint returnAddress(size_t offset) {
    assert(offset == 0);
    return StackPoint{this, StackPoint::ReturnAddress, offset};
  }

  StackPoint spilledRegister(size_t offset) {
    return StackPoint{this, StackPoint::SpilledRegisters, offset};
  }

  StackPoint calleeSavedRegister(size_t offset) {
    return StackPoint{this, StackPoint::CalleeSavedRegisters, offset};
  }

  StackPoint callerSavedRegister(size_t offset) {
    return StackPoint{this, StackPoint::CallerSavedRegisters, offset};
  }

  StackPoint callArgument(size_t offset) {
    return StackPoint{this, StackPoint::CallArguments, offset};
  }

private:
  size_t returnAddressSize = 0;
  size_t spilledRegistersSize = 0;
  size_t calleeSavedRegistersSize = 0;
  size_t callerSavedRegistersSize = 0;
  size_t callArgumentsSize = 0;
};

} // namespace kecc

#endif // KECC_TRANSLATE_FUNCTION_STACK_H
