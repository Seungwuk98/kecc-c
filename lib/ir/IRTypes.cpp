#include "kecc/ir/IRTypes.h"
#include "kecc/ir/Type.h"
#include "llvm/ADT/ArrayRef.h"

DEFINE_KECC_TYPE_ID(kecc::ir::IntT);
DEFINE_KECC_TYPE_ID(kecc::ir::FloatT);
DEFINE_KECC_TYPE_ID(kecc::ir::NameStruct);
DEFINE_KECC_TYPE_ID(kecc::ir::UnitT);
DEFINE_KECC_TYPE_ID(kecc::ir::FunctionT);
DEFINE_KECC_TYPE_ID(kecc::ir::PointerT);
DEFINE_KECC_TYPE_ID(kecc::ir::ConstQualifier);
DEFINE_KECC_TYPE_ID(kecc::ir::TupleT);

namespace kecc::ir {

class IntTImpl : public TypeImplTemplate<std::pair<int, int>> {
public:
  static IntTImpl *create(TypeStorage *storage, const KeyTy &key) {
    return create(storage, key.first, key.second);
  }

  static IntTImpl *create(TypeStorage *storage, int bitWidth, bool isSigned) {
    auto *impl =
        new (storage->allocate<IntTImpl>()) IntTImpl(bitWidth, isSigned);
    return impl;
  }

  int getBitWidth() const { return getKeyValue().first; }
  bool isSigned() const { return getKeyValue().second; }

private:
  IntTImpl(int bitWidth, int isSigned)
      : TypeImplTemplate({bitWidth, isSigned}) {}
};

class FloatTImpl : public TypeImplTemplate<int> {
public:
  static FloatTImpl *create(TypeStorage *storage, int bitWidth) {
    auto *impl = new (storage->allocate<FloatTImpl>()) FloatTImpl(bitWidth);
    return impl;
  }

  int getBitWidth() const { return getKeyValue(); }

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

  llvm::StringRef getName() const { return getKeyValue(); }

private:
  NameStructImpl(llvm::StringRef name) : TypeImplTemplate(name) {}
};

class FunctionTImpl
    : public TypeImplTemplate<
          std::pair<llvm::ArrayRef<Type>, llvm::ArrayRef<Type>>> {
public:
  static FunctionTImpl *create(TypeStorage *storage, const KeyTy &key) {
    return create(storage, key.first, key.second);
  }
  static FunctionTImpl *create(TypeStorage *storage,
                               llvm::ArrayRef<Type> returnTypes,
                               llvm::ArrayRef<Type> argTypes) {
    auto copiedReturnTypes = storage->copyArray(returnTypes);
    auto copiedArgTypes = storage->copyArray(argTypes);
    auto *impl = new (storage->allocate<FunctionTImpl>())
        FunctionTImpl({copiedReturnTypes, copiedArgTypes});
    return impl;
  }

  llvm::ArrayRef<Type> getReturnType() const { return getKeyValue().first; }
  llvm::ArrayRef<Type> getArgTypes() const { return getKeyValue().second; }

private:
  FunctionTImpl(llvm::ArrayRef<Type> returnType, llvm::ArrayRef<Type> argTypes)
      : TypeImplTemplate({returnType, argTypes}) {}
};

class PointerTImpl : public TypeImplTemplate<std::pair<Type, int>> {
public:
  static PointerTImpl *create(TypeStorage *storage, const KeyTy &key) {
    return create(storage, key.first, key.second);
  }

  static PointerTImpl *create(TypeStorage *storage, Type pointeeType,
                              bool isConst) {
    auto *impl = new (storage->allocate<PointerTImpl>())
        PointerTImpl(pointeeType, isConst);
    return impl;
  }
  Type getPointeeType() const { return getKeyValue().first; }
  bool isConst() const { return getKeyValue().second; }

private:
  PointerTImpl(Type pointeeType, bool isConst)
      : TypeImplTemplate({pointeeType, isConst}) {}
};

class ArrayTImpl : public TypeImplTemplate<std::pair<std::size_t, Type>> {
public:
  static ArrayTImpl *create(TypeStorage *storage, const KeyTy &key) {
    return create(storage, key.first, key.second);
  }

  static ArrayTImpl *create(TypeStorage *storage, std::size_t size,
                            Type elementType) {
    auto *impl =
        new (storage->allocate<ArrayTImpl>()) ArrayTImpl(size, elementType);
    return impl;
  }

  std::size_t getSize() const { return getKeyValue().first; }
  Type getElementType() const { return getKeyValue().second; }

private:
  ArrayTImpl(std::size_t size, Type elementType)
      : TypeImplTemplate({size, elementType}) {}
};

class ConstQualifierImpl : public TypeImplTemplate<Type> {
public:
  static ConstQualifierImpl *create(TypeStorage *storage, Type type) {
    auto *impl =
        new (storage->allocate<ConstQualifierImpl>()) ConstQualifierImpl(type);
    return impl;
  }
  Type getType() const { return getKeyValue(); }

private:
  ConstQualifierImpl(Type type) : TypeImplTemplate(type) {}
};

class TupleTImpl : public TypeImplTemplate<llvm::ArrayRef<Type>> {
public:
  static TupleTImpl *create(TypeStorage *storage, llvm::ArrayRef<Type> types) {
    auto copiedTypes = storage->copyArray(types);
    auto *impl = new (storage->allocate<TupleTImpl>()) TupleTImpl(copiedTypes);
    return impl;
  }

  llvm::ArrayRef<Type> getElementTypes() const { return getKeyValue(); }

private:
  TupleTImpl(llvm::ArrayRef<Type> types) : TypeImplTemplate(types) {}
};

IntT IntT::get(IRContext *context, int bitWidth, bool isSigned) {
  return Base::get(context, bitWidth, isSigned);
}

void IntT::printer(IntT type, llvm::raw_ostream &os) {
  os << (type.isSigned() ? "i" : "u") << type.getBitWidth();
}

bool IntT::isSigned() const { return getImpl()->isSigned(); }
int IntT::getBitWidth() const { return getImpl()->getBitWidth(); }

FloatT FloatT::get(IRContext *context, int bitWidth) {
  return Base::get(context, bitWidth);
}

void FloatT::printer(FloatT type, llvm::raw_ostream &os) {
  os << "f" << type.getBitWidth();
}

int FloatT::getBitWidth() const { return getImpl()->getBitWidth(); }

NameStruct NameStruct::get(IRContext *context, llvm::StringRef name) {
  return Base::get(context, name);
}

void NameStruct::printer(NameStruct type, llvm::raw_ostream &os) {
  os << "struct " << type.getName();
}

llvm::StringRef NameStruct::getName() const { return getImpl()->getName(); }

UnitT UnitT::get(IRContext *context) { return Base::get(context); }

void UnitT::printer(UnitT type, llvm::raw_ostream &os) { os << "unit"; }

FunctionT FunctionT::get(IRContext *context, llvm::ArrayRef<Type> returnTypes,
                         llvm::ArrayRef<Type> argTypes) {
  return Base::get(context, returnTypes, argTypes);
}

void FunctionT::printer(FunctionT type, llvm::raw_ostream &os) {
  os << "[ret:";

  for (auto I = type.getReturnTypes().begin(), E = type.getReturnTypes().end();
       I != E; ++I) {
    if (I != type.getReturnTypes().begin()) {
      os << ", ";
    }
    os << I->toString();
  }

  os << " params:(";

  for (auto I = type.getArgTypes().begin(), E = type.getArgTypes().end();
       I != E; ++I) {
    if (I != type.getArgTypes().begin()) {
      os << ", ";
    }
    os << I->toString();
  }

  os << ")]";
}

llvm::ArrayRef<Type> FunctionT::getReturnTypes() const {
  return getImpl()->getReturnType();
}
llvm::ArrayRef<Type> FunctionT::getArgTypes() const {
  return getImpl()->getArgTypes();
}

PointerT PointerT::get(IRContext *context, Type pointeeType, bool isConst) {
  return Base::get(context, pointeeType, isConst);
}

Type PointerT::getPointeeType() const { return getImpl()->getPointeeType(); }
bool PointerT::isConst() const { return getImpl()->isConst(); }

void PointerT::printer(PointerT type, llvm::raw_ostream &os) {
  os << type.getPointeeType().toString() << "*"
     << (type.isConst() ? "const" : "");
}

ArrayT ArrayT::get(IRContext *context, std::size_t size, Type elementType) {
  return Base::get(context, size, elementType);
}

Type ArrayT::getElementType() const { return getImpl()->getElementType(); }
std::size_t ArrayT::getSize() const { return getImpl()->getSize(); }
void ArrayT::printer(ArrayT type, llvm::raw_ostream &os) {
  os << "[" << type.getSize() << " x " << type.getElementType().toString()
     << "]";
}

ConstQualifier ConstQualifier::get(IRContext *context, Type type) {
  return Base::get(context, type);
}

void ConstQualifier::printer(ConstQualifier type, llvm::raw_ostream &os) {
  os << "const " << type.getType().toString();
}

Type ConstQualifier::getType() const { return getImpl()->getType(); }

TupleT TupleT::get(IRContext *context, llvm::ArrayRef<Type> elementTypes) {
  return Base::get(context, elementTypes);
}

void TupleT::printer(TupleT type, llvm::raw_ostream &os) {
  os << "(";
  for (auto I = type.getElementTypes().begin(),
            E = type.getElementTypes().end();
       I != E; ++I) {
    if (I != type.getElementTypes().begin()) {
      os << ", ";
    }
    os << I->toString();
  }
  os << ")";
}

llvm::ArrayRef<Type> TupleT::getElementTypes() const {
  return getImpl()->getElementTypes();
}

bool isCastableTo(Type from, Type to) {
  if (from.isa<ConstQualifier>())
    from = from.cast<ConstQualifier>().getType();

  if (to.isa<ConstQualifier>())
    to = to.cast<ConstQualifier>().getType();

  if (from == to)
    return true;

  if (from.isa<IntT>()) {
    if (to.isa<IntT, FloatT, PointerT>())
      return true;
  }

  if (from.isa<FloatT>()) {
    if (to.isa<IntT, FloatT>())
      return true;
  }

  if (from.isa<PointerT>()) {
    if (to.isa<PointerT, IntT>())
      return true;
  }

  return false;
}

} // namespace kecc::ir
