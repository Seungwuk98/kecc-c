#include "kecc/ir/InterpreterValue.h"
#include "llvm/Support/Error.h"

namespace kecc::ir {

void VInteger::createValue() {
  if (bitWidth == 1) {
    value = value & 1;
    isSignedV = false;
  } else if (bitWidth < 64) {
    if (isSignedV) {
      int shift = 64 - bitWidth;
      std::int64_t signedValue = value;
      signedValue = (signedValue << shift) >> shift;
      value = static_cast<std::uint64_t>(signedValue);
    } else {
      std::uint64_t mask = (1ULL << bitWidth) - 1;
      value = value & mask;
    }
  }
}

VInteger VInteger::operator+(const VInteger &other) const {
  assert(isSameType(other) && "Both integers must have the same type");
  std::uint64_t newValue = value + other.value;
  return VInteger(newValue, isSignedV, bitWidth);
}
VInteger VInteger::operator-(const VInteger &other) const {
  assert(isSameType(other) && "Both integers must have the same type");
  std::uint64_t newValue = value - other.value;
  return VInteger(newValue, isSignedV, bitWidth);
}

#define INT_OPERATE(op)                                                        \
  assert(isSameType(other) && "Both integers must have the same type");        \
  std::uint64_t newValue;                                                      \
  if (isSignedV) {                                                             \
    switch (bitWidth) {                                                        \
    case 8:                                                                    \
      newValue = static_cast<std::int8_t>(value)                               \
          op static_cast<std::int8_t>(other.value);                            \
      break;                                                                   \
    case 16:                                                                   \
      newValue = static_cast<std::int16_t>(value)                              \
          op static_cast<std::int16_t>(other.value);                           \
      break;                                                                   \
    case 32:                                                                   \
      newValue = static_cast<std::int32_t>(value)                              \
          op static_cast<std::int32_t>(other.value);                           \
      break;                                                                   \
    case 64:                                                                   \
      newValue = static_cast<std::int64_t>(value)                              \
          op static_cast<std::int64_t>(other.value);                           \
      break;                                                                   \
    default:                                                                   \
      llvm_unreachable("Unsupported integer bit width");                       \
    }                                                                          \
  } else {                                                                     \
    switch (bitWidth) {                                                        \
    case 8:                                                                    \
      newValue = static_cast<std::uint8_t>(value)                              \
          op static_cast<std::uint8_t>(other.value);                           \
      break;                                                                   \
    case 16:                                                                   \
      newValue = static_cast<std::uint16_t>(value)                             \
          op static_cast<std::uint16_t>(other.value);                          \
      break;                                                                   \
    case 32:                                                                   \
      newValue = static_cast<std::uint32_t>(value)                             \
          op static_cast<std::uint32_t>(other.value);                          \
      break;                                                                   \
    case 64:                                                                   \
      newValue = static_cast<std::uint64_t>(value)                             \
          op static_cast<std::uint64_t>(other.value);                          \
      break;                                                                   \
    default:                                                                   \
      llvm_unreachable("Unsupported integer bit width");                       \
    }                                                                          \
  }                                                                            \
  return VInteger(newValue, isSignedV, bitWidth)

VInteger VInteger::operator*(const VInteger &other) const { INT_OPERATE(*); }
llvm::Expected<VInteger> VInteger::operator/(const VInteger &other) const {
  if (other.value == 0) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Division by zero");
  }
  INT_OPERATE(/);
}
llvm::Expected<VInteger> VInteger::operator%(const VInteger &other) const {
  if (other.value == 0) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Modulo by zero");
  }
  INT_OPERATE(%);
}
VInteger VInteger::operator&(const VInteger &other) const {
  std::uint64_t newValue = value & other.value;
  return VInteger(newValue, isSignedV, bitWidth);
}
VInteger VInteger::operator|(const VInteger &other) const {
  std::uint64_t newValue = value | other.value;
  return VInteger(newValue, isSignedV, bitWidth);
}
VInteger VInteger::operator^(const VInteger &other) const {
  std::uint64_t newValue = value ^ other.value;
  return VInteger(newValue, isSignedV, bitWidth);
}
VInteger VInteger::operator<<(const VInteger &other) const { INT_OPERATE(<<); }
VInteger VInteger::operator>>(const VInteger &other) const { INT_OPERATE(>>); }

VInteger VInteger::operator==(const VInteger &other) const {
  assert(isSameType(other) && "Both integers must have the same type");
  return VInteger(value == other.value ? 1 : 0, false, 1);
}
VInteger VInteger::operator!=(const VInteger &other) const {
  assert(isSameType(other) && "Both integers must have the same type");
  return VInteger(value != other.value ? 1 : 0, false, 1);
}

#undef INT_OPERATE
#define INT_CMP_OPERATE(op)                                                    \
  assert(isSameType(other) && "Both integers must have the same type");        \
  bool result;                                                                 \
  if (isSignedV) {                                                             \
    switch (bitWidth) {                                                        \
    case 1:                                                                    \
    case 8:                                                                    \
      result = static_cast<std::int8_t>(value)                                 \
          op static_cast<std::int8_t>(other.value);                            \
      break;                                                                   \
    case 16:                                                                   \
      result = static_cast<std::int16_t>(value)                                \
          op static_cast<std::int16_t>(other.value);                           \
      break;                                                                   \
    case 32:                                                                   \
      result = static_cast<std::int32_t>(value)                                \
          op static_cast<std::int32_t>(other.value);                           \
      break;                                                                   \
    case 64:                                                                   \
      result = static_cast<std::int64_t>(value)                                \
          op static_cast<std::int64_t>(other.value);                           \
      break;                                                                   \
    default:                                                                   \
      llvm_unreachable("Unsupported integer bit width");                       \
    }                                                                          \
  } else {                                                                     \
    result = value op other.value;                                             \
  }                                                                            \
  return VInteger(result ? 1 : 0, false, 1)

VInteger VInteger::operator<(const VInteger &other) const {
  INT_CMP_OPERATE(<);
}
VInteger VInteger::operator<=(const VInteger &other) const {
  INT_CMP_OPERATE(<=);
}
VInteger VInteger::operator>(const VInteger &other) const {
  INT_CMP_OPERATE(>);
}
VInteger VInteger::operator>=(const VInteger &other) const {
  INT_CMP_OPERATE(>=);
}
#undef INT_CMP_OPERATE

VInteger VInteger::operator!() const {
  return VInteger(value == 0 ? 1 : 0, false, 1);
}
VInteger VInteger::operator~() const {
  return VInteger(~value, isSignedV, bitWidth);
}
VInteger VInteger::operator-() const {
  if (isSignedV) {
    switch (bitWidth) {
    case 1:
    case 8:
      return VInteger(
          static_cast<std::int8_t>(-static_cast<std::int8_t>(value)), isSignedV,
          bitWidth);
    case 16:
      return VInteger(
          static_cast<std::int16_t>(-static_cast<std::int16_t>(value)),
          isSignedV, bitWidth);
    case 32:
      return VInteger(-static_cast<std::int32_t>(value), isSignedV, bitWidth);
    case 64:
      return VInteger(-static_cast<std::int64_t>(value), isSignedV, bitWidth);
    default:
      llvm_unreachable("Unsupported integer bit width");
    }
  } else {
    // For unsigned integers, negation is equivalent to two's complement
    return VInteger(-value, isSignedV, bitWidth);
  }
}
VInteger VInteger::operator+() const { return *this; }

void VInteger::castFrom(const VInteger &other) {
  value = other.value;
  createValue();
}
void VInteger::castFrom(const VFloat &other) {
  if (other.bitWidth == 32) {
    if (isSignedV) {
      switch (bitWidth) {
      case 1:
        llvm::report_fatal_error("Cannot cast float to i1");
      case 8:
        value = static_cast<std::int8_t>(other.getAsFloat());
        break;
      case 16:
        value = static_cast<std::int16_t>(other.getAsFloat());
        break;
      case 32:
        value = static_cast<std::int32_t>(other.getAsFloat());
        break;
      case 64:
        value = static_cast<std::int64_t>(other.getAsFloat());
        break;
      default:
        llvm_unreachable("Unsupported integer bit width");
      }
    } else {
      switch (bitWidth) {
      case 1:
        llvm::report_fatal_error("Cannot cast float to i1");
      case 8:
        value = static_cast<std::uint8_t>(other.getAsFloat());
        break;
      case 16:
        value = static_cast<std::uint16_t>(other.getAsFloat());
        break;
      case 32:
        value = static_cast<std::uint32_t>(other.getAsFloat());
        break;
      case 64:
        value = static_cast<std::uint64_t>(other.getAsFloat());
        break;
      default:
        llvm_unreachable("Unsupported integer bit width");
      }
    }
  } else {
    if (isSignedV) {
      switch (bitWidth) {
      case 1:
        llvm::report_fatal_error("Cannot cast float to i1");
      case 8:
        value = static_cast<std::int8_t>(other.getAsDouble());
        break;
      case 16:
        value = static_cast<std::int16_t>(other.getAsDouble());
        break;
      case 32:
        value = static_cast<std::int32_t>(other.getAsDouble());
        break;
      case 64:
        value = static_cast<std::int64_t>(other.getAsDouble());
        break;
      default:
        llvm_unreachable("Unsupported integer bit width");
      }
    } else {
      switch (bitWidth) {
      case 1:
        llvm::report_fatal_error("Cannot cast float to i1");
      case 8:
        value = static_cast<std::uint8_t>(other.getAsDouble());
        break;
      case 16:
        value = static_cast<std::uint16_t>(other.getAsDouble());
        break;
      case 32:
        value = static_cast<std::uint32_t>(other.getAsDouble());
        break;
      case 64:
        value = static_cast<std::uint64_t>(other.getAsDouble());
        break;
      default:
        llvm_unreachable("Unsupported integer bit width");
      }
    }
  }
  createValue();
}

void VInteger::print(llvm::raw_ostream &os) const {
  if (isSignedV) {
    switch (bitWidth) {
    case 1:
      os << (value ? 1 : 0);
      break;
    case 8:
      os << static_cast<std::int32_t>(static_cast<std::int8_t>(value));
      break;
    case 16:
      os << static_cast<std::int16_t>(value);
      break;
    case 32:
      os << static_cast<std::int32_t>(value);
      break;
    case 64:
      os << static_cast<std::int64_t>(value);
      break;
    default:
      llvm_unreachable("Unsupported integer bit width");
    }
  } else {
    os << value;
  }
}
void VFloat::print(llvm::raw_ostream &os) const {
  if (bitWidth == 32) {
    os << getAsFloat();
  } else {
    os << getAsDouble();
  }
}

#define FLOAT_OPERATE(op)                                                      \
  assert(isSameType(other) && "Both floats must have the same type");          \
  std::uint64_t bitValue;                                                      \
  if (bitWidth == 32) {                                                        \
    float newValue = getAsFloat() op other.getAsFloat();                       \
    bitValue = llvm::bit_cast<std::uint32_t>(newValue);                        \
  } else {                                                                     \
    double newValue = getAsDouble() op other.getAsDouble();                    \
    bitValue = llvm::bit_cast<std::uint64_t>(newValue);                        \
  }                                                                            \
  return VFloat(bitValue, bitWidth)

VFloat VFloat::operator+(const VFloat &other) const { FLOAT_OPERATE(+); }
VFloat VFloat::operator-(const VFloat &other) const { FLOAT_OPERATE(-); }
VFloat VFloat::operator*(const VFloat &other) const { FLOAT_OPERATE(*); }
llvm::Expected<VFloat> VFloat::operator/(const VFloat &other) const {
  if (other.isZero()) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Division by zero");
  }
  FLOAT_OPERATE(/);
}
#undef FLOAT_OPERATE
#define FLOAT_CMP_OPERATE(op)                                                  \
  assert(isSameType(other) && "Both floats must have the same type");          \
  if (bitWidth == 32) {                                                        \
    return VInteger(getAsFloat() op other.getAsFloat() ? 1 : 0, false, 1);     \
  } else {                                                                     \
    return VInteger(getAsDouble() op other.getAsDouble() ? 1 : 0, false, 1);   \
  }

VInteger VFloat::operator<(const VFloat &other) const { FLOAT_CMP_OPERATE(<); }
VInteger VFloat::operator<=(const VFloat &other) const {
  FLOAT_CMP_OPERATE(<=);
}
VInteger VFloat::operator>(const VFloat &other) const { FLOAT_CMP_OPERATE(>); }
VInteger VFloat::operator>=(const VFloat &other) const {
  FLOAT_CMP_OPERATE(>=);
}
VInteger VFloat::operator==(const VFloat &other) const {
  FLOAT_CMP_OPERATE(==);
}
VInteger VFloat::operator!=(const VFloat &other) const {
  FLOAT_CMP_OPERATE(!=);
}
#undef FLOAT_CMP_OPERATE

VFloat VFloat::operator-() const {
  std::uint64_t bitValue;
  if (bitWidth == 32) {
    float newValue = -getAsFloat();
    bitValue = llvm::bit_cast<std::uint32_t>(newValue);
  } else {
    double newValue = -getAsDouble();
    bitValue = llvm::bit_cast<std::uint64_t>(newValue);
  }
  return VFloat(bitValue, bitWidth);
}
VFloat VFloat::operator+() const { return *this; }

VInteger VFloat::operator!() const {
  return VInteger(isZero() ? 1 : 0, false, 1);
}
bool VFloat::isZero() const {
  if (bitWidth == 32) {
    return getAsFloat() == 0.0f || getAsFloat() == -0.0f;
  } else {
    return getAsDouble() == 0.0 || getAsDouble() == -0.0;
  }
}

void VFloat::castFrom(const VFloat &other) {
  if (other.bitWidth == bitWidth) {
    value = other.value;
  } else if (other.bitWidth == 32 && bitWidth == 64) {
    // float to double
    float f = other.getAsFloat();
    double d = static_cast<double>(f);
    value = llvm::bit_cast<std::uint64_t>(d);
  } else if (other.bitWidth == 64 && bitWidth == 32) {
    // double to float
    double d = other.getAsDouble();
    float f = static_cast<float>(d);
    value = llvm::bit_cast<std::uint32_t>(f);
  } else {
    llvm_unreachable("Unsupported float bit width");
  }
}

float VFloat::getAsFloat() const {
  assert(bitWidth == 32 && "Bit width must be 32 for float");
  return llvm::bit_cast<float>(static_cast<std::uint32_t>(value));
}

double VFloat::getAsDouble() const {
  assert(bitWidth == 64 && "Bit width must be 64 for double");
  return llvm::bit_cast<double>(value);
}

VFloat VFloat::fromAPFloat(const llvm::APFloat &apFloat) {
  if (&apFloat.getSemantics() == &llvm::APFloat::IEEEsingle()) {
    float f = apFloat.convertToFloat();
    std::uint32_t bitValue = llvm::bit_cast<std::uint32_t>(f);
    return VFloat(bitValue, 32);
  } else if (&apFloat.getSemantics() == &llvm::APFloat::IEEEdouble()) {
    double d = apFloat.convertToDouble();
    std::uint64_t bitValue = llvm::bit_cast<std::uint64_t>(d);
    return VFloat(bitValue, 64);
  } else {
    llvm_unreachable("Unsupported APFloat semantics");
  }
}

void VFloat::castFrom(const VInteger &other) {
  if (bitWidth == 32) {
    if (other.isSignedV) {
      switch (other.bitWidth) {
      case 1:
        llvm::report_fatal_error("Cannot cast i1 to float");
      case 8:
        value = llvm::bit_cast<std::uint32_t>(
            static_cast<float>(static_cast<std::int8_t>(other.value)));
        break;
      case 16:
        value = llvm::bit_cast<std::uint32_t>(
            static_cast<float>(static_cast<std::int16_t>(other.value)));
        break;
      case 32:
        value = llvm::bit_cast<std::uint32_t>(
            static_cast<float>(static_cast<std::int32_t>(other.value)));
        break;
      case 64:
        value = llvm::bit_cast<std::uint32_t>(
            static_cast<float>(static_cast<std::int64_t>(other.value)));
        break;
      default:
        llvm_unreachable("Unsupported integer bit width");
      }
    } else {
      switch (other.bitWidth) {
      case 1:
        llvm::report_fatal_error("Cannot cast i1 to float");
      case 8:
        value = llvm::bit_cast<std::uint32_t>(
            static_cast<float>(static_cast<std::uint8_t>(other.value)));
        break;
      case 16:
        value = llvm::bit_cast<std::uint32_t>(
            static_cast<float>(static_cast<std::uint16_t>(other.value)));
        break;
      case 32:
        value = llvm::bit_cast<std::uint32_t>(
            static_cast<float>(static_cast<std::uint32_t>(other.value)));
        break;
      case 64:
        value = llvm::bit_cast<std::uint32_t>(
            static_cast<float>(static_cast<std::uint64_t>(other.value)));
        break;
      default:
        llvm_unreachable("Unsupported integer bit width");
      }
    }
  } else {
    if (other.isSignedV) {
      switch (other.bitWidth) {
      case 1:
        llvm::report_fatal_error("Cannot cast i1 to float");
      case 8:
        value = llvm::bit_cast<std::uint64_t>(
            static_cast<double>(static_cast<std::int8_t>(other.value)));
        break;
      case 16:
        value = llvm::bit_cast<std::uint64_t>(
            static_cast<double>(static_cast<std::int16_t>(other.value)));
        break;
      case 32:
        value = llvm::bit_cast<std::uint64_t>(
            static_cast<double>(static_cast<std::int32_t>(other.value)));
        break;
      case 64:
        value = llvm::bit_cast<std::uint64_t>(
            static_cast<double>(static_cast<std::int64_t>(other.value)));
        break;
      }
    } else {
      switch (other.bitWidth) {
      case 1:
        llvm::report_fatal_error("Cannot cast i1 to float");
      case 8:
        value = llvm::bit_cast<std::uint64_t>(
            static_cast<double>(static_cast<std::uint8_t>(other.value)));
        break;
      case 16:
        value = llvm::bit_cast<std::uint64_t>(
            static_cast<double>(static_cast<std::uint16_t>(other.value)));
        break;
      case 32:
        value = llvm::bit_cast<std::uint64_t>(
            static_cast<double>(static_cast<std::uint32_t>(other.value)));
        break;
      case 64:
        value = llvm::bit_cast<std::uint64_t>(
            static_cast<double>(static_cast<std::uint64_t>(other.value)));
        break;
      default:
        llvm_unreachable("Unsupported integer bit width");
      }
    }
  }
}

} // namespace kecc::ir
