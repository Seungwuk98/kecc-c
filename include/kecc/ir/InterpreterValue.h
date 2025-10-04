//===- kecc/ir/InterpreterValue.h - Virtual Value in Interpreter -*- C++ -*-==//
///
/// \file
/// This file declares a class to represent virtual integer and floating-point
/// values. IR Interpreter uses this class to perform operations on these values
/// instead of APInt, APFloat. It because APInt and APFloat doesn't simulate
/// perfectly every hardware.
///
//===----------------------------------------------------------------------===//

#ifndef KECC_IR_INTERPRETER_VALUE_H
#define KECC_IR_INTERPRETER_VALUE_H

#include "llvm/ADT/APFloat.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>

namespace kecc::ir {

class VFloat;

class VInteger {
public:
  VInteger(std::uint64_t value, bool isSigned, int bitWidth)
      : value(value), isSignedV(isSigned), bitWidth(bitWidth) {
    createValue();
  }
  VInteger(bool isSigned, int bitWidth)
      : value(0), isSignedV(isSigned), bitWidth(bitWidth) {
    if (bitWidth == 1)
      isSignedV = false;
  }

  VInteger operator+(const VInteger &other) const;
  VInteger operator-(const VInteger &other) const;
  VInteger operator*(const VInteger &other) const;
  llvm::Expected<VInteger> operator/(const VInteger &other) const;
  llvm::Expected<VInteger> operator%(const VInteger &other) const;
  VInteger operator&(const VInteger &other) const;
  VInteger operator|(const VInteger &other) const;
  VInteger operator^(const VInteger &other) const;
  VInteger operator<<(const VInteger &other) const;
  VInteger operator>>(const VInteger &other) const;
  VInteger operator~() const;
  VInteger operator-() const;
  VInteger operator==(const VInteger &other) const;
  VInteger operator!=(const VInteger &other) const;
  VInteger operator<(const VInteger &other) const;
  VInteger operator<=(const VInteger &other) const;
  VInteger operator>(const VInteger &other) const;
  VInteger operator>=(const VInteger &other) const;
  VInteger operator!() const;
  VInteger operator+() const;

  void print(llvm::raw_ostream &os) const;

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const VInteger &val) {
    val.print(os);
    return os;
  }

  std::uint64_t getValue() const { return value; }
  bool isSigned() const { return isSignedV; }

  void castFrom(const VInteger &other);
  void castFrom(const VFloat &other);

private:
  friend class VFloat;
  bool isSameType(const VInteger &other) const {
    if (bitWidth == 1 && other.bitWidth == 1)
      return true;
    return bitWidth == other.bitWidth && isSignedV == other.isSignedV;
  }

  void createValue();

  std::uint64_t value;
  bool isSignedV;
  int bitWidth;
};

class VFloat {
public:
  VFloat(double value)
      : value(std::bit_cast<std::uint64_t>(value)), bitWidth(64) {}
  VFloat(float value)
      : value(std::bit_cast<std::uint32_t>(value)), bitWidth(32) {}
  VFloat(int bitWidth) : value(0), bitWidth(bitWidth) {
    assert(bitWidth == 32 || bitWidth == 64);
  }
  VFloat(std::uint64_t value, int bitWidth) : value(value), bitWidth(bitWidth) {
    assert(bitWidth == 32 || bitWidth == 64);
  }

  VFloat operator+(const VFloat &other) const;
  VFloat operator-(const VFloat &other) const;
  VFloat operator*(const VFloat &other) const;
  llvm::Expected<VFloat> operator/(const VFloat &other) const;
  VFloat operator-() const;
  VInteger operator==(const VFloat &other) const;
  VInteger operator!=(const VFloat &other) const;
  VInteger operator<(const VFloat &other) const;
  VInteger operator<=(const VFloat &other) const;
  VInteger operator>(const VFloat &other) const;
  VInteger operator>=(const VFloat &other) const;
  VInteger operator!() const;
  VFloat operator+() const;

  std::uint64_t getRawValue() const { return value; }
  float getAsFloat() const;
  double getAsDouble() const;

  void print(llvm::raw_ostream &os) const;
  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const VFloat &val) {
    val.print(os);
    return os;
  }

  bool isZero() const;

  void castFrom(const VInteger &other);
  void castFrom(const VFloat &other);
  static VFloat fromAPFloat(const llvm::APFloat &apFloat);

private:
  bool isSameType(const VFloat &other) const {
    return bitWidth == other.bitWidth;
  }

  friend class VInteger;
  std::uint64_t value;
  int bitWidth;
};

} // namespace kecc::ir

#endif // KECC_IR_INTERPRETER_VALUE_H
