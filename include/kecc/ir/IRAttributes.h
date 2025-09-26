#ifndef KECC_IR_IR_ATTRIBUTES_H
#define KECC_IR_IR_ATTRIBUTES_H

#include "kecc/ir/Attribute.h"
#include "kecc/ir/Block.h"
#include "kecc/ir/Context.h"
#include "kecc/ir/IRTypes.h"
#include "kecc/ir/Instruction.h"
#include "kecc/ir/Type.h"
#include "kecc/ir/WalkSupport.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APSInt.h"

namespace kecc::ir {

class StringAttrImpl;
class StringAttr
    : public Attribute::Base<StringAttr, Attribute, StringAttrImpl> {
public:
  using Base::Base;
  static StringAttr get(IRContext *context, llvm::StringRef value);

  llvm::StringRef getValue() const;
};

class ConstantAttr : public Attribute {
public:
  using Attribute::Attribute;

  ConstantAttr insertMinus() const;
  Type getType() const;

  static bool classof(Attribute attr);
};

class ConstantIntAttrImpl;
class ConstantIntAttr : public Attribute::Base<ConstantIntAttr, ConstantAttr,
                                               ConstantIntAttrImpl> {
public:
  using Base::Base;
  static ConstantIntAttr get(IRContext *context, uint64_t value, int bitwidth,
                             bool isSigned);

  uint64_t getValue() const;
  int getBitWidth() const;
  bool isSigned() const;
  IntT getIntType() const;
  llvm::APInt getAsAPInt() const {
    return llvm::APInt(getBitWidth(), getValue(), isSigned());
  }
  llvm::APSInt getAsAPSInt() const {
    return llvm::APSInt(getAsAPInt(), !isSigned());
  }
};

class ArrayAttrImpl;
class ArrayAttr : public Attribute::Base<ArrayAttr, Attribute, ArrayAttrImpl> {
public:
  using Base::Base;
  static ArrayAttr get(IRContext *context, llvm::ArrayRef<Attribute> values);

  llvm::ArrayRef<Attribute> getValues() const;
};

class ConstantFloatAttrImpl;
class ConstantFloatAttr
    : public Attribute::Base<ConstantFloatAttr, ConstantAttr,
                             ConstantFloatAttrImpl> {
public:
  using Base::Base;
  static ConstantFloatAttr get(IRContext *context, double value);
  static ConstantFloatAttr get(IRContext *context, float value);
  static ConstantFloatAttr get(IRContext *context, llvm::APFloat value);

  FloatT getFloatType() const;

  llvm::APFloat getValue() const;
};

class ConstantStringFloatAttrImpl;
class ConstantStringFloatAttr
    : public Attribute::Base<ConstantStringFloatAttr, ConstantAttr,
                             ConstantStringFloatAttrImpl> {
public:
  using Base::Base;
  static ConstantStringFloatAttr get(IRContext *context, llvm::StringRef value,
                                     FloatT floatType);

  llvm::StringRef getValue() const;
  FloatT getFloatType() const;
  ConstantFloatAttr convertToFloatAttr() const;
};

class ConstantUndefAttrImpl;
class ConstantUndefAttr
    : public Attribute::Base<ConstantUndefAttr, ConstantAttr,
                             ConstantUndefAttrImpl> {
public:
  using Base::Base;
  static ConstantUndefAttr get(IRContext *context, Type type);

  Type getUndefType() const;
};

class ConstantUnitAttr
    : public Attribute::Base<ConstantUnitAttr, ConstantAttr, AttributeImpl> {
public:
  using Base::Base;

  static ConstantUnitAttr get(IRContext *context);
};

class ConstantVariableAttrImpl;
class ConstantVariableAttr
    : public Attribute::Base<ConstantVariableAttr, ConstantAttr,
                             ConstantVariableAttrImpl> {
public:
  using Base::Base;
  static ConstantVariableAttr get(IRContext *context, llvm::StringRef name,
                                  Type type);

  llvm::StringRef getName() const;
  Type getVariableType() const;
};

class TypeAttrImpl;
class TypeAttr : public Attribute::Base<TypeAttr, Attribute, TypeAttrImpl> {
public:
  using Base::Base;
  static TypeAttr get(IRContext *context, Type type);

  Type getType() const;
};

class EnumAttrImpl;
class EnumAttr : public Attribute::Base<EnumAttr, Attribute, EnumAttrImpl> {
public:
  using Base::Base;

  template <typename T> static EnumAttr get(IRContext *context, T value) {
    TypeID id = TypeID::get<T>();
    return get(context, id, static_cast<unsigned>(value));
  }

  template <typename T> T getEnumValue() const {
    return static_cast<T>(getEnumValueAsUnsigned());
  }

  static EnumAttr get(IRContext *context, TypeID typeId, unsigned value);

  unsigned getEnumValueAsUnsigned() const;

private:
};

using RangeKey = std::pair<intptr_t, intptr_t>;
RangeKey convertToRangeKey(const llvm::SMRange &range);
llvm::SMRange convertToRange(const RangeKey &key);

class InitializerAttr : public Attribute {
public:
  using Attribute::Attribute;

  Attribute interpret() const;
  void printInitializer(IRPrintContext &context) const;
  llvm::SMRange getRange() const;

  static bool classof(Attribute attr);
};

class ASTInitializerListImpl;
class ASTInitializerList
    : public Attribute::Base<ASTInitializerList, InitializerAttr,
                             ASTInitializerListImpl> {
public:
  using Base::Base;

  static ASTInitializerList get(IRContext *context, RangeKey range,
                                llvm::ArrayRef<InitializerAttr> values) {
    return get(context, convertToRange(range), values);
  }
  static ASTInitializerList get(IRContext *context, llvm::SMRange range,
                                llvm::ArrayRef<InitializerAttr> values);

  llvm::SMRange getRange() const;
  llvm::ArrayRef<InitializerAttr> getValues() const;
  InitializerAttr getValue(size_t index) const;
  size_t size() const { return getValues().size(); }
  bool empty() const { return getValues().empty(); }

  ArrayAttr interpret() const;
};

class ASTGroupOpImpl;
class ASTGroupOp
    : public Attribute::Base<ASTGroupOp, InitializerAttr, ASTGroupOpImpl> {
public:
  using Base::Base;

  static ASTGroupOp get(IRContext *context, RangeKey range,
                        InitializerAttr value) {
    return get(context, convertToRange(range), value);
  }

  static ASTGroupOp get(IRContext *context, llvm::SMRange range,
                        InitializerAttr value);

  llvm::SMRange getRange() const;
  InitializerAttr getValue() const;

  ConstantAttr interpret() const;
};

class ASTUnaryOpImpl;
class ASTUnaryOp
    : public Attribute::Base<ASTUnaryOp, InitializerAttr, ASTUnaryOpImpl> {
public:
  enum OpKind { Plus, Minus };
  using Base::Base;

  static ASTUnaryOp get(IRContext *context, RangeKey range, OpKind kind,
                        InitializerAttr operand) {
    return get(context, convertToRange(range), kind, operand);
  }
  static ASTUnaryOp get(IRContext *context, llvm::SMRange range, OpKind kind,
                        InitializerAttr operand);

  llvm::SMRange getRange() const;
  OpKind getOpKind() const;
  InitializerAttr getOperand() const;
  ConstantAttr interpret() const;
};

class ASTIntegerImpl;
class ASTInteger
    : public Attribute::Base<ASTInteger, InitializerAttr, ASTIntegerImpl> {
public:
  enum IntegerBase { Decimal, Hexadecimal, Binary, Octal };
  enum Suffix {
    Long_L, // L
    Long_l, // l
    Int,
  };
  using Base::Base;

  static ASTInteger get(IRContext *context, RangeKey range, IntegerBase base,
                        llvm::StringRef value, Suffix suffix) {
    return get(context, convertToRange(range), base, value, suffix);
  }
  static ASTInteger get(IRContext *context, llvm::SMRange range,
                        IntegerBase base, llvm::StringRef value, Suffix suffix);

  llvm::SMRange getRange() const;
  IntegerBase getBase() const;
  llvm::StringRef getValue() const;
  Suffix getSuffix() const;
  ConstantIntAttr interpret() const;
};

class ASTFloatImpl;
class ASTFloat
    : public Attribute::Base<ASTFloat, InitializerAttr, ASTFloatImpl> {
public:
  enum Suffix { Float_F, Float_f, Double };
  using Base::Base;

  static ASTFloat get(IRContext *context, RangeKey range, llvm::StringRef value,
                      Suffix suffix) {
    return get(context, convertToRange(range), value, suffix);
  }
  static ASTFloat get(IRContext *context, llvm::SMRange range,
                      llvm::StringRef value, Suffix suffix);

  llvm::SMRange getRange() const;
  llvm::StringRef getValue() const;
  Suffix getSuffix() const;
  ConstantFloatAttr interpret() const;
};

void registerBuiltinAttributes(IRContext *context);

} // namespace kecc::ir

namespace llvm {

template <>
struct DenseMapInfo<kecc::ir::ASTUnaryOp::OpKind>
    : public DenseMapInfo<unsigned> {};

template <>
struct DenseMapInfo<kecc::ir::ASTInteger::IntegerBase>
    : public DenseMapInfo<unsigned> {};

template <>
struct DenseMapInfo<kecc::ir::ASTInteger::Suffix>
    : public DenseMapInfo<unsigned> {};

template <>
struct DenseMapInfo<kecc::ir::ASTFloat::Suffix>
    : public DenseMapInfo<unsigned> {};

} // namespace llvm

DECLARE_KECC_TYPE_ID(kecc::ir::StringAttr)
DECLARE_KECC_TYPE_ID(kecc::ir::ConstantIntAttr)
DECLARE_KECC_TYPE_ID(kecc::ir::ConstantFloatAttr)
DECLARE_KECC_TYPE_ID(kecc::ir::ConstantUndefAttr)
DECLARE_KECC_TYPE_ID(kecc::ir::ConstantVariableAttr)
DECLARE_KECC_TYPE_ID(kecc::ir::ArrayAttr)
DECLARE_KECC_TYPE_ID(kecc::ir::TypeAttr)
DECLARE_KECC_TYPE_ID(kecc::ir::EnumAttr)
DECLARE_KECC_TYPE_ID(kecc::ir::ASTInitializerList)
DECLARE_KECC_TYPE_ID(kecc::ir::ASTUnaryOp)
DECLARE_KECC_TYPE_ID(kecc::ir::ASTInteger)
DECLARE_KECC_TYPE_ID(kecc::ir::ASTFloat)

#endif // KECC_IR_IR_ATTRIBUTES_H
