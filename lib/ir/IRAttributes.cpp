#include "kecc/ir/IRAttributes.h"
#include "kecc/ir/Attribute.h"
#include "kecc/ir/Context.h"
#include "kecc/ir/Type.h"
#include "kecc/ir/Value.h"
#include "kecc/utils/LLVM.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/SMLoc.h"

DEFINE_KECC_TYPE_ID(kecc::ir::StringAttr)
DEFINE_KECC_TYPE_ID(kecc::ir::ConstantIntAttr)
DEFINE_KECC_TYPE_ID(kecc::ir::ConstantFloatAttr)
DEFINE_KECC_TYPE_ID(kecc::ir::ConstantUndefAttr)
DEFINE_KECC_TYPE_ID(kecc::ir::ConstantVariableAttr)
DEFINE_KECC_TYPE_ID(kecc::ir::ArrayAttr)
DEFINE_KECC_TYPE_ID(kecc::ir::TypeAttr)
DEFINE_KECC_TYPE_ID(kecc::ir::EnumAttr)
DEFINE_KECC_TYPE_ID(kecc::ir::ASTInitializerList)
DEFINE_KECC_TYPE_ID(kecc::ir::ASTUnaryOp)
DEFINE_KECC_TYPE_ID(kecc::ir::ASTInteger)
DEFINE_KECC_TYPE_ID(kecc::ir::ASTFloat)

namespace kecc::ir {

//==------------------------------------------------------------------------==//
/// String Attribute
//==------------------------------------------------------------------------==//

class StringAttrImpl : public AttributeImplTemplate<llvm::StringRef> {
public:
  static StringAttrImpl *create(TypeStorage *storage, llvm::StringRef value) {
    auto copiedValue = storage->copyString(value);
    auto *impl =
        new (storage->allocate<StringAttrImpl>()) StringAttrImpl(copiedValue);
    return impl;
  }

  llvm::StringRef getValue() const { return getKeyValue(); }

private:
  StringAttrImpl(llvm::StringRef value) : AttributeImplTemplate(value) {}
};

StringAttr StringAttr::get(IRContext *context, llvm::StringRef value) {
  return Base::get(context, value);
}

llvm::StringRef StringAttr::getValue() const { return getImpl()->getValue(); }

//==------------------------------------------------------------------------==//
/// Array Attribute
//==------------------------------------------------------------------------==//

class ArrayAttrImpl : public AttributeImplTemplate<llvm::ArrayRef<Attribute>> {
public:
  static ArrayAttrImpl *create(TypeStorage *storage,
                               llvm::ArrayRef<Attribute> values) {
    auto copiedValues = storage->copyArray(values);
    auto *impl =
        new (storage->allocate<ArrayAttrImpl>()) ArrayAttrImpl(copiedValues);
    return impl;
  }

  llvm::ArrayRef<Attribute> getValues() const { return getKeyValue(); }

private:
  ArrayAttrImpl(llvm::ArrayRef<Attribute> values)
      : AttributeImplTemplate(values) {}
};

ArrayAttr ArrayAttr::get(IRContext *context, llvm::ArrayRef<Attribute> values) {
  return Base::get(context, values);
}

llvm::ArrayRef<Attribute> ArrayAttr::getValues() const {
  return getImpl()->getValues();
}

//==------------------------------------------------------------------------==//
/// Constant
//==------------------------------------------------------------------------==//

bool ConstantAttr::classof(Attribute attr) {
  return attr.isa<ConstantIntAttr, ConstantFloatAttr, ConstantStringFloatAttr,
                  ConstantUndefAttr, ConstantUnitAttr, ConstantVariableAttr>();
}

Type ConstantAttr::getType() const {
  return llvm::TypeSwitch<Attribute, Type>(*this)
      .Case([&](ConstantIntAttr attr) -> Type { return attr.getIntType(); })
      .Case([&](ConstantFloatAttr attr) { return attr.getFloatType(); })
      .Case([&](ConstantStringFloatAttr attr) { return attr.getFloatType(); })
      .Case([&](ConstantUndefAttr attr) { return attr.getUndefType(); })
      .Case([&](ConstantVariableAttr attr) { return attr.getVariableType(); })
      .Case([&](ConstantUnitAttr attr) { return UnitT::get(getContext()); })
      .Default([&](Attribute) { return nullptr; });
}

ConstantAttr ConstantAttr::insertMinus() const {
  return llvm::TypeSwitch<ConstantAttr, ConstantAttr>(*this)
      .Case([&](ConstantIntAttr intAttr) {
        return ConstantIntAttr::get(getContext(), -intAttr.getValue(),
                                    intAttr.getBitWidth(), intAttr.isSigned());
      })
      .Case([&](ConstantFloatAttr floatAttr) {
        llvm::APFloat negatedValue = -floatAttr.getValue();
        return ConstantFloatAttr::get(getContext(), negatedValue);
      })
      .Case([&](ConstantUndefAttr undefAttr) { return nullptr; })
      .Case([&](ConstantVariableAttr varAttr) { return nullptr; })
      .Default([&](ConstantAttr) -> ConstantAttr {
        llvm_unreachable("Unsupported Constant Attribute type for insertMinus");
      });
}

//==------------------------------------------------------------------------==//
/// Constant Int
//==------------------------------------------------------------------------==//

class ConstantIntAttrImpl
    : public AttributeImplTemplate<std::tuple<std::uint64_t, int, int>> {
public:
  static ConstantIntAttrImpl *create(TypeStorage *storage, const KeyTy &key) {
    return create(storage, std::get<0>(key), std::get<1>(key),
                  std::get<2>(key));
  }

  static ConstantIntAttrImpl *create(TypeStorage *storage, uint64_t value,
                                     int bitwidth, bool isSigned) {
    auto *impl = new (storage->allocate<ConstantIntAttrImpl>())
        ConstantIntAttrImpl(value, bitwidth, isSigned);
    return impl;
  }

  uint64_t getValue() const { return std::get<0>(getKeyValue()); }
  int getBitWidth() const { return std::get<1>(getKeyValue()); }
  bool isSigned() const {
    assert(std::get<2>(getKeyValue()) == 0 ||
           std::get<2>(getKeyValue()) == 1 && "isSigned must be either 0 or 1");
    return std::get<2>(getKeyValue());
  }

private:
  ConstantIntAttrImpl(std::uint64_t value, int bitwidth, int isSigned)
      : AttributeImplTemplate({value, bitwidth, isSigned}) {}
};

ConstantIntAttr ConstantIntAttr::get(IRContext *context, uint64_t value,
                                     int bitwidth, bool isSigned) {
  if (bitwidth < 64) {
    if (bitwidth == 1) {
      value = value ? 1 : 0;
    } else if (!isSigned)
      value = value & ((1ULL << bitwidth) - 1);
    else {
      value = value << (64 - bitwidth);
      value = static_cast<int64_t>(value) >> (64 - bitwidth);
    }
  }
  return Base::get(context, value, bitwidth, isSigned);
}

std::uint64_t ConstantIntAttr::getValue() const {
  return getImpl()->getValue();
}
int ConstantIntAttr::getBitWidth() const { return getImpl()->getBitWidth(); }
bool ConstantIntAttr::isSigned() const { return getImpl()->isSigned(); }

IntT ConstantIntAttr::getIntType() const {
  return IntT::get(getContext(), getBitWidth(), isSigned());
}

//==------------------------------------------------------------------------==//
/// Constant Float
//==------------------------------------------------------------------------==//

class ConstantFloatAttrImpl : public AttributeImplTemplate<llvm::APFloat> {
public:
  static ConstantFloatAttrImpl *create(TypeStorage *storage,
                                       llvm::APFloat value) {
    auto *impl = new (storage->allocate<ConstantFloatAttrImpl>())
        ConstantFloatAttrImpl(value);
    return impl;
  }

  llvm::APFloat getValue() const { return getKeyValue(); }

private:
  ConstantFloatAttrImpl(llvm::APFloat value) : AttributeImplTemplate(value) {}
};

ConstantFloatAttr ConstantFloatAttr::get(IRContext *context, double value) {
  llvm::APFloat apFloat(value);
  return Base::get(context, apFloat);
}

ConstantFloatAttr ConstantFloatAttr::get(IRContext *context, float value) {
  llvm::APFloat apFloat(value);
  return Base::get(context, apFloat);
}

ConstantFloatAttr ConstantFloatAttr::get(IRContext *context,
                                         llvm::APFloat value) {
  return Base::get(context, value);
}

llvm::APFloat ConstantFloatAttr::getValue() const {
  return getImpl()->getValue();
}

FloatT ConstantFloatAttr::getFloatType() const {
  if (&getValue().getSemantics() == &llvm::APFloat::IEEEsingle())
    return FloatT::get(getContext(), 32);
  else if (&getValue().getSemantics() == &llvm::APFloat::IEEEdouble())
    return FloatT::get(getContext(), 64);
  else
    llvm_unreachable("Unsupported float type for ConstantFloatAttr");
}

class ConstantStringFloatAttrImpl
    : public AttributeImplTemplate<std::pair<llvm::StringRef, FloatT>> {
public:
  static ConstantStringFloatAttrImpl *create(TypeStorage *storage,
                                             const KeyTy &key) {
    return create(storage, key.first, key.second);
  }

  static ConstantStringFloatAttrImpl *
  create(TypeStorage *storage, llvm::StringRef value, FloatT floatType) {
    auto copiedValue = storage->copyString(value);
    auto *impl = new (storage->allocate<ConstantStringFloatAttrImpl>())
        ConstantStringFloatAttrImpl(copiedValue, floatType);
    return impl;
  }

  llvm::StringRef getValue() const { return getKeyValue().first; }
  FloatT getFloatType() const { return getKeyValue().second; }

  ConstantFloatAttr convertToFloatAttr() const {
    if (getFloatType().getBitWidth() == 32) {
      llvm::APFloat apFloat(llvm::APFloat::IEEEsingle());
      auto err = apFloat.convertFromString(getValue(),
                                           llvm::APFloat::rmNearestTiesToEven);
      assert(err && "Failed to convert string to APFloat");
      (void)err;
      return ConstantFloatAttr::get(getContext(), apFloat);
    } else if (getFloatType().getBitWidth() == 64) {
      llvm::APFloat apFloat(llvm::APFloat::IEEEdouble());
      auto err = apFloat.convertFromString(getValue(),
                                           llvm::APFloat::rmNearestTiesToEven);
      assert(err && "Failed to convert string to APFloat");
      return ConstantFloatAttr::get(getContext(), apFloat);
    } else {
      llvm_unreachable("Unsupported float type for ConstantStringFloat");
    }
  }

private:
  ConstantStringFloatAttrImpl(llvm::StringRef value, FloatT floatType)
      : AttributeImplTemplate({value, floatType}) {}
};

ConstantStringFloatAttr ConstantStringFloatAttr::get(IRContext *context,
                                                     llvm::StringRef value,
                                                     FloatT floatType) {
  return Base::get(context, value, floatType);
}

llvm::StringRef ConstantStringFloatAttr::getValue() const {
  return getImpl()->getValue();
}
FloatT ConstantStringFloatAttr::getFloatType() const {
  return getImpl()->getFloatType();
}
ConstantFloatAttr ConstantStringFloatAttr::convertToFloatAttr() const {
  return getImpl()->convertToFloatAttr();
}

//==------------------------------------------------------------------------==//
/// Constant Undef
//==------------------------------------------------------------------------==//

class ConstantUndefAttrImpl : public AttributeImplTemplate<Type> {
public:
  static ConstantUndefAttrImpl *create(TypeStorage *storage, Type type) {
    auto *impl = new (storage->allocate<ConstantUndefAttrImpl>())
        ConstantUndefAttrImpl(type);
    return impl;
  }

  Type getType() const { return getKeyValue(); }

private:
  ConstantUndefAttrImpl(Type type) : AttributeImplTemplate(type) {}
};

ConstantUndefAttr ConstantUndefAttr::get(IRContext *context, Type type) {
  return Base::get(context, type);
}

Type ConstantUndefAttr::getUndefType() const { return getImpl()->getType(); }

//==------------------------------------------------------------------------==//
/// Constant Undef
//==------------------------------------------------------------------------==//

ConstantUnitAttr ConstantUnitAttr::get(IRContext *context) {
  return Base::get(context);
}

//==------------------------------------------------------------------------==//
/// Constant Variable
//==------------------------------------------------------------------------==//

class ConstantVariableAttrImpl
    : public AttributeImplTemplate<std::pair<llvm::StringRef, Type>> {
public:
  static ConstantVariableAttrImpl *create(TypeStorage *storage,
                                          const KeyTy &key) {
    return create(storage, key.first, key.second);
  }

  static ConstantVariableAttrImpl *create(TypeStorage *storage,
                                          llvm::StringRef name, Type type) {
    auto copiedName = storage->copyString(name);
    auto *impl = new (storage->allocate<ConstantVariableAttrImpl>())
        ConstantVariableAttrImpl(copiedName, type);
    return impl;
  }

  llvm::StringRef getName() const { return getKeyValue().first; }
  Type getVariableType() const { return getKeyValue().second; }

private:
  ConstantVariableAttrImpl(llvm::StringRef name, Type type)
      : AttributeImplTemplate({name, type}) {}
};

ConstantVariableAttr
ConstantVariableAttr::get(IRContext *context, llvm::StringRef name, Type type) {
  return Base::get(context, name, type);
}

llvm::StringRef ConstantVariableAttr::getName() const {
  return getImpl()->getName();
}

Type ConstantVariableAttr::getVariableType() const {
  return getImpl()->getVariableType();
}

//==------------------------------------------------------------------------==//
/// Type Attribute
//==------------------------------------------------------------------------==//

class TypeAttrImpl : public AttributeImplTemplate<Type> {
public:
  static TypeAttrImpl *create(TypeStorage *storage, Type type) {
    auto *impl = new (storage->allocate<TypeAttrImpl>()) TypeAttrImpl(type);
    return impl;
  }

  Type getType() const { return getKeyValue(); }

private:
  TypeAttrImpl(Type type) : AttributeImplTemplate(type) {}
};

TypeAttr TypeAttr::get(IRContext *context, Type type) {
  return Base::get(context, type);
}

Type TypeAttr::getType() const { return getImpl()->getType(); }

//==------------------------------------------------------------------------==//
/// Enum Attribute
//==------------------------------------------------------------------------==//

class EnumAttrImpl : public AttributeImplTemplate<std::pair<TypeID, unsigned>> {
public:
  static EnumAttrImpl *create(TypeStorage *storage, const KeyTy &key) {
    return create(storage, key.first, key.second);
  }

  static EnumAttrImpl *create(TypeStorage *storage, TypeID typeId,
                              unsigned value) {
    auto *impl =
        new (storage->allocate<EnumAttrImpl>()) EnumAttrImpl(typeId, value);
    return impl;
  }

  TypeID getId() const { return getKeyValue().first; }
  unsigned getEnumValueAsUnsigned() const { return getKeyValue().second; }

private:
  EnumAttrImpl(TypeID id, unsigned value)
      : AttributeImplTemplate({id, value}) {}
};

EnumAttr EnumAttr::get(IRContext *context, TypeID typeId, unsigned value) {
  return Base::get(context, typeId, value);
}

unsigned EnumAttr::getEnumValueAsUnsigned() const {
  return getImpl()->getEnumValueAsUnsigned();
}

//==------------------------------------------------------------------------==//
/// Initializer
//==------------------------------------------------------------------------==//

bool InitializerAttr::classof(Attribute attr) {
  return attr.isa<ASTInitializerList, ASTUnaryOp, ASTInteger, ASTFloat>();
}

Attribute InitializerAttr::interpret() const {
  return llvm::TypeSwitch<Attribute, Attribute>(*this)
      .Case<ASTInitializerList>([&](ASTInitializerList initializerList) {
        return initializerList.interpret();
      })
      .Case<ASTUnaryOp>([&](ASTUnaryOp unaryOp) { return unaryOp.interpret(); })
      .Case<ASTInteger>([&](ASTInteger integer) { return integer.interpret(); })
      .Case<ASTFloat>([&](ASTFloat floatAttr) { return floatAttr.interpret(); })
      .Case<ASTGroupOp>(
          [&](ASTGroupOp groupAttr) { return groupAttr.interpret(); })
      .Default([&](Attribute) { return nullptr; });
}

RangeKey convertToRangeKey(const llvm::SMRange &range) {
  return {reinterpret_cast<intptr_t>(range.Start.getPointer()),
          reinterpret_cast<intptr_t>(range.End.getPointer())};
}

llvm::SMRange convertToRange(const RangeKey &key) {
  return {
      llvm::SMLoc::getFromPointer(reinterpret_cast<const char *>(key.first)),
      llvm::SMLoc::getFromPointer(reinterpret_cast<const char *>(key.second))};
}

void InitializerAttr::printInitializer(IRPrintContext &context) const {
  llvm::TypeSwitch<Attribute>(*this)
      .Case([&](ASTInitializerList initializerList) {
        context.getOS() << "{";
        for (size_t i = 0; i < initializerList.size(); ++i) {
          if (i > 0)
            context.getOS() << ", ";
          initializerList.getValue(i).printInitializer(context);
        }
        context.getOS() << "}";
      })
      .Case([&](ASTUnaryOp unaryOp) {
        context.getOS() << (unaryOp.getOpKind() == ASTUnaryOp::Plus ? "+"
                                                                    : "-");
        unaryOp.getOperand().printInitializer(context);
      })
      .Case([&](ASTInteger integer) {
        switch (integer.getBase()) {
        case ASTInteger::Decimal:
          break;
        case ASTInteger::Hexadecimal:
          context.getOS() << "0x";
          break;
        case ASTInteger::Binary:
          context.getOS() << "0b";
          break;
        case ASTInteger::Octal:
          context.getOS() << "0";
          break;
        }

        context.getOS() << integer.getValue();
        if (integer.getSuffix() == ASTInteger::Long_L)
          context.getOS() << "L";
        else if (integer.getSuffix() == ASTInteger::Long_l)
          context.getOS() << "l";
      })
      .Case([&](ASTGroupOp groupAttr) {
        context.getOS() << '(';
        groupAttr.getValue().printInitializer(context);
        context.getOS() << ')';
      })
      .Case([&](ASTFloat floatAttr) {
        context.getOS() << floatAttr.getValue();
        if (floatAttr.getSuffix() == ASTFloat::Float_f)
          context.getOS() << "f";
        else if (floatAttr.getSuffix() == ASTFloat::Float_F)
          context.getOS() << "F";
      })
      .Default([&](Attribute) {
        llvm_unreachable("Unsupported Initializer Attribute type for printing");
      });
}

llvm::SMRange InitializerAttr::getRange() const {
  return llvm::TypeSwitch<Attribute, llvm::SMRange>(*this)
      .Case([&](ASTInitializerList initializerList) {
        return initializerList.getRange();
      })
      .Case([&](ASTUnaryOp unaryOp) { return unaryOp.getRange(); })
      .Case([&](ASTInteger integer) { return integer.getRange(); })
      .Case([&](ASTFloat floatAttr) { return floatAttr.getRange(); })
      .Case([&](ASTGroupOp groupAttr) { return groupAttr.getRange(); })
      .Default([&](Attribute) -> llvm::SMRange {
        llvm_unreachable("Unsupported Initializer Attribute type for getRange");
      });
}

//==------------------------------------------------------------------------==//
/// AST Initializer List
//==------------------------------------------------------------------------==//

class ASTInitializerListImpl
    : public AttributeImplTemplate<
          std::pair<RangeKey, llvm::ArrayRef<InitializerAttr>>> {
public:
  static KeyTy getKey(llvm::SMRange range,
                      llvm::ArrayRef<InitializerAttr> values) {
    return {convertToRangeKey(range), values};
  }

  static ASTInitializerListImpl *create(TypeStorage *storage,
                                        const KeyTy &key) {
    auto range = convertToRange(key.first);
    return create(storage, range, key.second);
  }

  static ASTInitializerListImpl *
  create(TypeStorage *storage, llvm::SMRange range,
         llvm::ArrayRef<InitializerAttr> values) {
    auto copiedValues = storage->copyArray(values);
    auto *impl = new (storage->allocate<ASTInitializerListImpl>())
        ASTInitializerListImpl(convertToRangeKey(range), copiedValues);
    return impl;
  }

  llvm::SMRange getRange() const { return convertToRange(getKeyValue().first); }

  llvm::ArrayRef<InitializerAttr> getValues() const {
    return getKeyValue().second;
  }

private:
  ASTInitializerListImpl(RangeKey key, llvm::ArrayRef<InitializerAttr> values)
      : AttributeImplTemplate({key, values}) {}
};

ASTInitializerList
ASTInitializerList::get(IRContext *context, llvm::SMRange range,
                        llvm::ArrayRef<InitializerAttr> values) {
  return Base::get(context, range, values);
}

llvm::SMRange ASTInitializerList::getRange() const {
  return getImpl()->getRange();
}

llvm::ArrayRef<InitializerAttr> ASTInitializerList::getValues() const {
  return getImpl()->getValues();
}

InitializerAttr ASTInitializerList::getValue(size_t index) const {
  assert(index < size() && "Index out of bounds");
  return getValues()[index];
}

ArrayAttr ASTInitializerList::interpret() const {
  llvm::SmallVector<Attribute> interpretedValues;
  interpretedValues.reserve(size());

  for (auto operand : getValues()) {
    auto interpretedValue = operand.interpret();
    if (!interpretedValue)
      return nullptr;

    interpretedValues.emplace_back(interpretedValue);
  }

  return ArrayAttr::get(getContext(), interpretedValues);
}

//==------------------------------------------------------------------------==//
/// AST Group
//==------------------------------------------------------------------------==//

class ASTGroupOpImpl
    : public AttributeImplTemplate<std::pair<RangeKey, InitializerAttr>> {
public:
  static KeyTy getKey(llvm::SMRange range, InitializerAttr value) {
    return {convertToRangeKey(range), value};
  }

  static ASTGroupOpImpl *create(TypeStorage *storage, const KeyTy &key) {
    return create(storage, convertToRange(key.first), key.second);
  }

  static ASTGroupOpImpl *create(TypeStorage *storage, llvm::SMRange range,
                                InitializerAttr value) {
    auto *impl =
        new (storage->allocate<ASTGroupOpImpl>()) ASTGroupOpImpl(range, value);
    return impl;
  }

  llvm::SMRange getRange() const {
    return convertToRange(std::get<0>(getKeyValue()));
  }
  InitializerAttr getValue() const { return std::get<1>(getKeyValue()); }

private:
  ASTGroupOpImpl(llvm::SMRange range, InitializerAttr value)
      : AttributeImplTemplate({convertToRangeKey(range), value}) {}
};

ASTGroupOp ASTGroupOp::get(IRContext *context, llvm::SMRange range,
                           InitializerAttr value) {
  return Base::get(context, range, value);
}

llvm::SMRange ASTGroupOp::getRange() const { return getImpl()->getRange(); }

InitializerAttr ASTGroupOp::getValue() const { return getImpl()->getValue(); }

ConstantAttr ASTGroupOp::interpret() const {
  auto interpretedValue = getValue().interpret();
  assert((interpretedValue.isa<ConstantAttr>()) &&
         "ASTGroupOp value must be interpretable to a ConstantAttr");
  return interpretedValue.cast<ConstantAttr>();
}

//==------------------------------------------------------------------------==//
/// AST Unary Operation
//==------------------------------------------------------------------------==//

class ASTUnaryOpImpl
    : public AttributeImplTemplate<
          std::tuple<RangeKey, ASTUnaryOp::OpKind, InitializerAttr>> {
public:
  static KeyTy getKey(llvm::SMRange range, ASTUnaryOp::OpKind kind,
                      InitializerAttr value) {
    return {convertToRangeKey(range), kind, value};
  }

  static ASTUnaryOpImpl *create(TypeStorage *storage, const KeyTy &key) {
    return create(storage, convertToRange(std::get<0>(key)), std::get<1>(key),
                  std::get<2>(key));
  }

  static ASTUnaryOpImpl *create(TypeStorage *storage, llvm::SMRange range,
                                ASTUnaryOp::OpKind kind,
                                InitializerAttr value) {
    auto *impl = new (storage->allocate<ASTUnaryOpImpl>())
        ASTUnaryOpImpl(convertToRangeKey(range), kind, value);
    return impl;
  }

  llvm::SMRange getRange() const {
    return convertToRange(std::get<0>(getKeyValue()));
  }
  ASTUnaryOp::OpKind getOpKind() const { return std::get<1>(getKeyValue()); }
  InitializerAttr getOperand() const { return std::get<2>(getKeyValue()); }

private:
  ASTUnaryOpImpl(RangeKey key, ASTUnaryOp::OpKind kind, InitializerAttr value)
      : AttributeImplTemplate({key, kind, value}) {}
};

ASTUnaryOp ASTUnaryOp::get(IRContext *context, llvm::SMRange range,
                           ASTUnaryOp::OpKind kind, InitializerAttr value) {
  return Base::get(context, range, kind, value);
}

llvm::SMRange ASTUnaryOp::getRange() const { return getImpl()->getRange(); }
ASTUnaryOp::OpKind ASTUnaryOp::getOpKind() const {
  return getImpl()->getOpKind();
}
InitializerAttr ASTUnaryOp::getOperand() const {
  return getImpl()->getOperand();
}

ConstantAttr ASTUnaryOp::interpret() const {
  auto interpretedOperand = getOperand().interpret();
  return llvm::TypeSwitch<Attribute, ConstantAttr>(interpretedOperand)
      .Case<ConstantIntAttr>([&](ConstantIntAttr operand) {
        if (getOpKind() == ASTUnaryOp::Plus)
          return operand; // Unary plus has no effect on the value.

        return ConstantIntAttr::get(getContext(), -operand.getValue(),
                                    operand.getBitWidth(), operand.isSigned());
      })
      .Case<ConstantFloatAttr>([&](ConstantFloatAttr operand) {
        if (getOpKind() == ASTUnaryOp::Plus)
          return operand;
        llvm::APFloat negatedValue = -operand.getValue();
        return ConstantFloatAttr::get(getContext(), negatedValue);
      })
      .Default([&](Attribute) { return nullptr; });
}

//==------------------------------------------------------------------------==//
/// AST Integer
//==------------------------------------------------------------------------==//

class ASTIntegerImpl : public AttributeImplTemplate<
                           std::tuple<RangeKey, ASTInteger::IntegerBase,
                                      llvm::StringRef, ASTInteger::Suffix>> {
public:
  static ASTIntegerImpl *create(TypeStorage *storage, const KeyTy &key) {
    return create(storage, convertToRange(std::get<0>(key)),
                  static_cast<ASTInteger::IntegerBase>(std::get<1>(key)),
                  std::get<2>(key),
                  static_cast<ASTInteger::Suffix>(std::get<3>(key)));
  }

  static ASTIntegerImpl *create(TypeStorage *storage, llvm::SMRange range,
                                ASTInteger::IntegerBase base,
                                llvm::StringRef value,
                                ASTInteger::Suffix suffix) {
    return new (storage->allocate<ASTIntegerImpl>())
        ASTIntegerImpl(convertToRangeKey(range), base, value, suffix);
  }

  static KeyTy getKey(llvm::SMRange range, ASTInteger::IntegerBase base,
                      llvm::StringRef value, ASTInteger::Suffix suffix) {
    return {convertToRangeKey(range), base, value, suffix};
  }

  llvm::SMRange getRange() const {
    return convertToRange(std::get<0>(getKeyValue()));
  }
  ASTInteger::IntegerBase getBase() const {
    return static_cast<ASTInteger::IntegerBase>(std::get<1>(getKeyValue()));
  }
  llvm::StringRef getValue() const { return std::get<2>(getKeyValue()); }
  ASTInteger::Suffix getSuffix() const {
    return static_cast<ASTInteger::Suffix>(std::get<3>(getKeyValue()));
  }

private:
  ASTIntegerImpl(RangeKey key, ASTInteger::IntegerBase base,
                 llvm::StringRef value, ASTInteger::Suffix suffix)
      : AttributeImplTemplate({key, base, value, suffix}) {}
};

ASTInteger ASTInteger::get(IRContext *context, llvm::SMRange range,
                           ASTInteger::IntegerBase base, llvm::StringRef value,
                           ASTInteger::Suffix suffix) {
  return Base::get(context, range, base, value, suffix);
}

llvm::SMRange ASTInteger::getRange() const { return getImpl()->getRange(); }
ASTInteger::IntegerBase ASTInteger::getBase() const {
  return getImpl()->getBase();
}
llvm::StringRef ASTInteger::getValue() const { return getImpl()->getValue(); }
ASTInteger::Suffix ASTInteger::getSuffix() const {
  return getImpl()->getSuffix();
}

ConstantIntAttr ASTInteger::interpret() const {
  // Convert the string value to an integer based on the base and suffix.

  std::uint64_t value;
  bool succeed = false;
  switch (getBase()) {
  case ASTInteger::Decimal:
    succeed = getValue().getAsInteger(10, value);
    break;
  case ASTInteger::Hexadecimal:
    succeed = getValue().getAsInteger(16, value);
    break;
  case ASTInteger::Binary:
    succeed = getValue().getAsInteger(2, value);
    break;
  case ASTInteger::Octal:
    succeed = getValue().getAsInteger(8, value);
    break;
  };

  if (!succeed)
    return nullptr;

  // Determine the bit width and signedness based on the suffix.
  switch (getSuffix()) {
  case ASTInteger::Long_L:
  case ASTInteger::Long_l:
    return ConstantIntAttr::get(getContext(), value, 64, true);
  case ASTInteger::Int:
    return ConstantIntAttr::get(getContext(), value, 32, true);
  }
}

//==------------------------------------------------------------------------==//
/// AST Float
//==------------------------------------------------------------------------==//

class ASTFloatImpl
    : public AttributeImplTemplate<
          std::tuple<RangeKey, llvm::StringRef, ASTFloat::Suffix>> {
public:
  static KeyTy getKey(llvm::SMRange range, llvm::StringRef value,
                      ASTFloat::Suffix suffix) {
    return {convertToRangeKey(range), value, suffix};
  }

  static ASTFloatImpl *create(TypeStorage *storage, const KeyTy &key) {
    return create(storage, convertToRange(std::get<0>(key)), std::get<1>(key),
                  static_cast<ASTFloat::Suffix>(std::get<2>(key)));
  }

  static ASTFloatImpl *create(TypeStorage *storage, llvm::SMRange range,
                              llvm::StringRef value, ASTFloat::Suffix suffix) {
    auto *impl = new (storage->allocate<ASTFloatImpl>())
        ASTFloatImpl(convertToRangeKey(range), value, suffix);
    return impl;
  }

  llvm::SMRange getRange() const {
    return convertToRange(std::get<0>(getKeyValue()));
  }
  llvm::StringRef getValue() const { return std::get<1>(getKeyValue()); }
  ASTFloat::Suffix getSuffix() const {
    return static_cast<ASTFloat::Suffix>(std::get<2>(getKeyValue()));
  }

private:
  ASTFloatImpl(RangeKey key, llvm::StringRef value, ASTFloat::Suffix suffix)
      : AttributeImplTemplate({key, value, suffix}) {}
};

ASTFloat ASTFloat::get(IRContext *context, llvm::SMRange range,
                       llvm::StringRef value, ASTFloat::Suffix suffix) {
  return Base::get(context, range, value, suffix);
}

llvm::SMRange ASTFloat::getRange() const { return getImpl()->getRange(); }
llvm::StringRef ASTFloat::getValue() const { return getImpl()->getValue(); }
ASTFloat::Suffix ASTFloat::getSuffix() const { return getImpl()->getSuffix(); }

ConstantFloatAttr ASTFloat::interpret() const {
  llvm::APFloat floatValue(getSuffix() == ASTFloat::Double
                               ? llvm::APFloat::IEEEdouble()
                               : llvm::APFloat::IEEEsingle());
  auto err = floatValue.convertFromString(getValue(),
                                          llvm::APFloat::rmNearestTiesToEven);
  if (!err.takeError().success())
    return nullptr;

  return ConstantFloatAttr::get(getContext(), floatValue);
}

void registerBuiltinAttributes(IRContext *context) {
  context->registerAttr<StringAttr>();
  context->registerAttr<ConstantIntAttr>();
  context->registerAttr<ConstantFloatAttr>();
  context->registerAttr<ConstantStringFloatAttr>();
  context->registerAttr<ConstantUndefAttr>();
  context->registerAttr<ConstantUnitAttr>();
  context->registerAttr<ConstantVariableAttr>();
  context->registerAttr<ArrayAttr>();
  context->registerAttr<TypeAttr>();
  context->registerAttr<EnumAttr>();
  context->registerAttr<ASTInitializerList>();
  context->registerAttr<ASTGroupOp>();
  context->registerAttr<ASTUnaryOp>();
  context->registerAttr<ASTInteger>();
  context->registerAttr<ASTFloat>();
}

} // namespace kecc::ir
