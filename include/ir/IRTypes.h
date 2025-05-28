#ifndef KECC_IR_IR_TYPES_H
#define KECC_IR_IR_TYPES_H

#include "ir/Context.h"
#include "ir/Types.h"
#include "utils/TypeId.h"

namespace kecc::ir {

class IntTImpl;
class FloatTImpl;
class NameStructImpl;
class FunctionTImpl;
class PointerTImpl;
class ConstQualifierImpl;

class IntT : public TypeTemplate<IntT, Type, IntTImpl> {
public:
  static IntT get(IRContext *context, int bitWidth, bool isSigned);
  static void printer(IntT type, llvm::raw_ostream &os);

  bool isSigned() const;
  int getBitWidth() const;
};

class FloatT : public TypeTemplate<FloatT, Type, FloatTImpl> {
public:
  static FloatT get(IRContext *context, int bitWidth);
  static void printer(FloatT type, llvm::raw_ostream &os);

  int getBitWidth() const;
};

class NameStruct : public TypeTemplate<NameStruct, Type, NameStructImpl> {
public:
  static NameStruct get(IRContext *context, llvm::StringRef name);
  static void printer(NameStruct type, llvm::raw_ostream &os);

  llvm::StringRef getName() const;
};

class UnitT : public TypeTemplate<UnitT, Type, TypeImpl> {
public:
  static UnitT get(IRContext *context);
  static void printer(UnitT type, llvm::raw_ostream &os);
};

class FunctionT : public TypeTemplate<FunctionT, Type, FunctionTImpl> {
public:
  static FunctionT get(IRContext *context, Type returnType,
                       llvm::ArrayRef<Type> argTypes);
  static void printer(FunctionT type, llvm::raw_ostream &os);

  Type getReturnType() const;
  llvm::ArrayRef<Type> getArgTypes() const;
};

class PointerT : public TypeTemplate<PointerT, Type, PointerTImpl> {
public:
  static PointerT get(IRContext *context, Type pointeeType);
  static void printer(PointerT type, llvm::raw_ostream &os);

  Type getPointeeType() const;
};

class ConstQualifier
    : public TypeTemplate<ConstQualifier, Type, ConstQualifierImpl> {
public:
  static ConstQualifier get(IRContext *context, Type type);
  static void printer(ConstQualifier type, llvm::raw_ostream &os);

  Type getType() const;
};

} // namespace kecc::ir

DECLARE_KECC_TYPE_ID(kecc::ir::IntT);
DECLARE_KECC_TYPE_ID(kecc::ir::FloatT);
DECLARE_KECC_TYPE_ID(kecc::ir::NameStruct);
DECLARE_KECC_TYPE_ID(kecc::ir::UnitT);
DECLARE_KECC_TYPE_ID(kecc::ir::FunctionT);
DECLARE_KECC_TYPE_ID(kecc::ir::PointerT);
DECLARE_KECC_TYPE_ID(kecc::ir::ConstQualifier);

#endif // KECC_IR_IR_TYPES_H
