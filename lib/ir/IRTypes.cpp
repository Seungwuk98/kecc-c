#include "ir/IRTypes.h"
#include "utils/TypeId.h"
#include "llvm/ADT/ArrayRef.h"

DEFINE_KECC_TYPE_ID(kecc::ir::IntT);
DEFINE_KECC_TYPE_ID(kecc::ir::FloatT);
DEFINE_KECC_TYPE_ID(kecc::ir::NameStruct);
DEFINE_KECC_TYPE_ID(kecc::ir::UnitT);
DEFINE_KECC_TYPE_ID(kecc::ir::FunctionT);
DEFINE_KECC_TYPE_ID(kecc::ir::PointerT);
DEFINE_KECC_TYPE_ID(kecc::ir::ConstQualifier);

namespace kecc::ir {

class IntTImpl : public TypeImplTemplate<std::pair<int, bool>> {
public:
  static IntTImpl *create(TypeStorage *storage, int bitWidth, bool isSigned) {
    auto *impl =
        new (storage->allocate<IntTImpl>()) IntTImpl(bitWidth, isSigned);
    return impl;
  }

  int getBitWidth() const { return getKey().first; }
  bool isSigned() const { return getKey().second; }

private:
  IntTImpl(int bitWidth, bool isSigned)
      : TypeImplTemplate({bitWidth, isSigned}) {}
};

class FloatTImpl : public TypeImplTemplate<int> {
public:
  static FloatTImpl *create(TypeStorage *storage, int bitWidth) {
    auto *impl = new (storage->allocate<FloatTImpl>()) FloatTImpl(bitWidth);
    return impl;
  }

  int getBitWidth() const { return getKey(); }

private:
  FloatTImpl(int bitWidth) : TypeImplTemplate(bitWidth) {}
};

class NameStructImpl : public TypeImplTemplate<llvm::StringRef> {
public:
  static NameStructImpl *create(TypeStorage *storage, llvm::StringRef name) {
    auto copiedName = storage->copyString(name);
    auto *impl =
        new (storage->allocate<NameStructImpl>()) NameStructImpl(copiedName);
    return impl;
  }

  llvm::StringRef getName() const { return getKey(); }

private:
  NameStructImpl(llvm::StringRef name) : TypeImplTemplate(name) {}
};

class FunctionTImpl
    : public TypeImplTemplate<std::pair<Type, llvm::ArrayRef<Type>>> {
public:
  static FunctionTImpl *create(TypeStorage *storage, Type returnType,
                               llvm::ArrayRef<Type> argTypes) {
    auto copiedArgTypes = storage->copyArray(argTypes);
    auto *impl = new (storage->allocate<FunctionTImpl>())
        FunctionTImpl({returnType, copiedArgTypes});
    return impl;
  }

  Type getReturnType() const { return getKey().first; }
  llvm::ArrayRef<Type> getArgTypes() const { return getKey().second; }

private:
  FunctionTImpl(Type returnType, llvm::ArrayRef<Type> argTypes)
      : TypeImplTemplate({returnType, argTypes}) {}
};

class PointerTImpl : public TypeImplTemplate<Type> {
public:
  static PointerTImpl *create(TypeStorage *storage, Type pointeeType) {
    auto *impl =
        new (storage->allocate<PointerTImpl>()) PointerTImpl(pointeeType);
    return impl;
  }
  Type getPointeeType() const { return getKey(); }

private:
  PointerTImpl(Type pointeeType) : TypeImplTemplate(pointeeType) {}
};

class ConstQualifierImpl : public TypeImplTemplate<Type> {
public:
  static ConstQualifierImpl *create(TypeStorage *storage, Type type) {
    auto *impl =
        new (storage->allocate<ConstQualifierImpl>()) ConstQualifierImpl(type);
    return impl;
  }
  Type getType() const { return getKey(); }

private:
  ConstQualifierImpl(Type type) : TypeImplTemplate(type) {}
};

IntT IntT::get(IRContext *context, int bitWidth, bool isSigned) {
  return Base::get(context, bitWidth, isSigned);
}

void IntT::printer(IntT type, llvm::raw_ostream &os) {
  os << (type.isSigned() ? "i" : "u") << type.getBitWidth();
}

FloatT FloatT::get(IRContext *context, int bitWidth) {
  return Base::get(context, bitWidth);
}

void FloatT::printer(FloatT type, llvm::raw_ostream &os) {
  os << "f" << type.getBitWidth();
}

NameStruct NameStruct::get(IRContext *context, llvm::StringRef name) {
  return Base::get(context, name);
}

void NameStruct::printer(NameStruct type, llvm::raw_ostream &os) {
  os << "struct " << type.getName();
}

UnitT UnitT::get(IRContext *context) { return Base::get(context); }

void UnitT::printer(UnitT type, llvm::raw_ostream &os) { os << "unit"; }

FunctionT FunctionT::get(IRContext *context, Type returnType,
                         llvm::ArrayRef<Type> argTypes) {
  return Base::get(context, returnType, argTypes);
}

void FunctionT::printer(FunctionT type, llvm::raw_ostream &os) {
  os << "[ret: " << type.getReturnType().toString() << ", params: (";

  for (auto I = type.getArgTypes().begin(), E = type.getArgTypes().end();
       I != E; ++I) {
    if (I != type.getArgTypes().begin()) {
      os << ", ";
    }
    os << I->toString();
  }

  os << ")]";
}

Type FunctionT::getReturnType() const { return getImpl()->getReturnType(); }
llvm::ArrayRef<Type> FunctionT::getArgTypes() const {
  return getImpl()->getArgTypes();
}

PointerT PointerT::get(IRContext *context, Type pointeeType) {
  return Base::get(context, pointeeType);
}

Type PointerT::getPointeeType() const { return getImpl()->getPointeeType(); }

void PointerT::printer(PointerT type, llvm::raw_ostream &os) {
  os << type.getPointeeType().toString() << "*";
}

ConstQualifier ConstQualifier::get(IRContext *context, Type type) {
  return Base::get(context, type);
}

void ConstQualifier::printer(ConstQualifier type, llvm::raw_ostream &os) {
  os << "const " << type.getType().toString();
}

Type ConstQualifier::getType() const { return getImpl()->getType(); }

} // namespace kecc::ir
