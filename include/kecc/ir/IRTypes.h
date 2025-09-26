#ifndef KECC_IR_IR_TYPES_H
#define KECC_IR_IR_TYPES_H

#include "kecc/ir/Context.h"
#include "kecc/ir/Type.h"
#include "kecc/parser/Lexer.h"
#include <cstddef>

namespace kecc::ir {

class IntTImpl;
class FloatTImpl;
class NameStructImpl;
class FunctionTImpl;
class PointerTImpl;
class ArrayTImpl;
class ConstQualifierImpl;
class TupleTImpl;

class IntT : public Type::Base<IntT, Type, IntTImpl> {
public:
  using Base::Base;
  static IntT get(IRContext *context, int bitWidth, bool isSigned);
  static void printer(IntT type, llvm::raw_ostream &os);
  static std::pair<size_t, size_t>
  calculateSizeAndAlign(IntT type, const StructSizeMap &sizeMap);

  static IntT getBool(IRContext *context) { return get(context, 1, true); }

  bool isSigned() const;
  int getBitWidth() const;
};

class FloatT : public Type::Base<FloatT, Type, FloatTImpl> {
public:
  using Base::Base;
  static FloatT get(IRContext *context, int bitWidth);
  static void printer(FloatT type, llvm::raw_ostream &os);
  static std::pair<size_t, size_t>
  calculateSizeAndAlign(FloatT type, const StructSizeMap &sizeMap);

  bool isF32() const { return getBitWidth() == 32; }
  bool isF64() const { return getBitWidth() == 64; }

  int getBitWidth() const;
};

class NameStruct : public Type::Base<NameStruct, Type, NameStructImpl> {
public:
  using Base::Base;
  static NameStruct get(IRContext *context, llvm::StringRef name);
  static void printer(NameStruct type, llvm::raw_ostream &os);
  static std::pair<size_t, size_t>
  calculateSizeAndAlign(NameStruct type, const StructSizeMap &sizeMap);

  llvm::StringRef getName() const;
};

class UnitT : public Type::Base<UnitT, Type, TypeImpl> {
public:
  using Base::Base;
  static UnitT get(IRContext *context);
  static void printer(UnitT type, llvm::raw_ostream &os);
  static std::pair<size_t, size_t> calculateSizeAndAlign(UnitT type,
                                                         const StructSizeMap &);
};

class FunctionT : public Type::Base<FunctionT, Type, FunctionTImpl> {
public:
  using Base::Base;
  static FunctionT get(IRContext *context, llvm::ArrayRef<Type> returnTypes,
                       llvm::ArrayRef<Type> argTypes);
  static void printer(FunctionT type, llvm::raw_ostream &os);
  static std::pair<size_t, size_t>
  calculateSizeAndAlign(FunctionT type, const StructSizeMap &sizeMap);

  llvm::ArrayRef<Type> getReturnTypes() const;
  llvm::ArrayRef<Type> getArgTypes() const;
};

class PointerT : public Type::Base<PointerT, Type, PointerTImpl> {
public:
  using Base::Base;
  static PointerT get(IRContext *context, Type pointeeType,
                      bool isConst = false);
  static void printer(PointerT type, llvm::raw_ostream &os);
  static std::pair<size_t, size_t>
  calculateSizeAndAlign(PointerT type, const StructSizeMap &sizeMap);

  Type getPointeeType() const;
  bool isConst() const;
};

class ArrayT : public Type::Base<ArrayT, Type, ArrayTImpl> {
public:
  using Base::Base;
  static ArrayT get(IRContext *context, std::size_t size, Type elementType);
  static void printer(ArrayT type, llvm::raw_ostream &os);
  static std::pair<size_t, size_t>
  calculateSizeAndAlign(ArrayT type, const StructSizeMap &sizeMap);
  std::size_t getSize() const;
  Type getElementType() const;
};

class ConstQualifier
    : public Type::Base<ConstQualifier, Type, ConstQualifierImpl> {
public:
  using Base::Base;
  static ConstQualifier get(IRContext *context, Type type);
  static void printer(ConstQualifier type, llvm::raw_ostream &os);
  static std::pair<size_t, size_t>
  calculateSizeAndAlign(ConstQualifier type, const StructSizeMap &sizeMap);

  Type getType() const;
};

class TupleT : public Type::Base<TupleT, Type, TupleTImpl> {
public:
  using Base::Base;
  static TupleT get(IRContext *context, llvm::ArrayRef<Type> elementTypes);
  static void printer(TupleT type, llvm::raw_ostream &os);
  static std::pair<size_t, size_t>
  calculateSizeAndAlign(TupleT type, const StructSizeMap &sizeMap);

  llvm::ArrayRef<Type> getElementTypes() const;
};

bool isCastableTo(Type from, Type to);

void registerBuiltinTypes(IRContext *context);

} // namespace kecc::ir

DECLARE_KECC_TYPE_ID(kecc::ir::IntT);
DECLARE_KECC_TYPE_ID(kecc::ir::FloatT);
DECLARE_KECC_TYPE_ID(kecc::ir::NameStruct);
DECLARE_KECC_TYPE_ID(kecc::ir::UnitT);
DECLARE_KECC_TYPE_ID(kecc::ir::FunctionT);
DECLARE_KECC_TYPE_ID(kecc::ir::PointerT);
DECLARE_KECC_TYPE_ID(kecc::ir::ConstQualifier);
DECLARE_KECC_TYPE_ID(kecc::ir::TupleT);

#endif // KECC_IR_IR_TYPES_H
