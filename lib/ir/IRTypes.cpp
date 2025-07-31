#include "kecc/ir/IRTypes.h"
#include "kecc/ir/Attribute.h"
#include "kecc/ir/Context.h"
#include "kecc/ir/Type.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include <cstddef>

DEFINE_KECC_TYPE_ID(kecc::ir::IntT);
DEFINE_KECC_TYPE_ID(kecc::ir::FloatT);
DEFINE_KECC_TYPE_ID(kecc::ir::NameStruct);
DEFINE_KECC_TYPE_ID(kecc::ir::UnitT);
DEFINE_KECC_TYPE_ID(kecc::ir::FunctionT);
DEFINE_KECC_TYPE_ID(kecc::ir::PointerT);
DEFINE_KECC_TYPE_ID(kecc::ir::ConstQualifier);
DEFINE_KECC_TYPE_ID(kecc::ir::TupleT);

namespace kecc::ir {

//===----------------------------------------------------------------------===//
// IntT
//===----------------------------------------------------------------------===//

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

IntT IntT::get(IRContext *context, int bitWidth, bool isSigned) {
  return Base::get(context, bitWidth, isSigned);
}

void IntT::printer(IntT type, llvm::raw_ostream &os) {
  os << (type.isSigned() ? "i" : "u") << type.getBitWidth();
}

bool IntT::isSigned() const { return getImpl()->isSigned(); }
int IntT::getBitWidth() const { return getImpl()->getBitWidth(); }

std::pair<size_t, size_t> IntT::calculateSizeAndAlign(IntT type,
                                                      const StructSizeMap &) {
  auto size = (type.getBitWidth() + BITS_OF_BYTE - 1) / BITS_OF_BYTE;
  auto align = size;
  return {size, align};
}

//===----------------------------------------------------------------------===//
// FloatT
//===----------------------------------------------------------------------===//

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
FloatT FloatT::get(IRContext *context, int bitWidth) {
  return Base::get(context, bitWidth);
}

void FloatT::printer(FloatT type, llvm::raw_ostream &os) {
  os << "f" << type.getBitWidth();
}

int FloatT::getBitWidth() const { return getImpl()->getBitWidth(); }

std::pair<size_t, size_t> FloatT::calculateSizeAndAlign(FloatT type,
                                                        const StructSizeMap &) {
  auto size = (type.getBitWidth() + BITS_OF_BYTE - 1) / BITS_OF_BYTE;
  auto align = size;
  return {size, align};
}

//===----------------------------------------------------------------------===//
// NameStruct
//===----------------------------------------------------------------------===//

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

NameStruct NameStruct::get(IRContext *context, llvm::StringRef name) {
  return Base::get(context, name);
}

void NameStruct::printer(NameStruct type, llvm::raw_ostream &os) {
  os << "struct " << type.getName();
}

llvm::StringRef NameStruct::getName() const { return getImpl()->getName(); }

std::pair<size_t, size_t>
NameStruct::calculateSizeAndAlign(NameStruct type,
                                  const StructSizeMap &sizeMap) {
  assert(sizeMap.contains(type.getName()) &&
         "NameStruct type must have a size defined in the size map");
  auto [size, align, _] = sizeMap.at(type.getName());
  return {size, align};
}

//===----------------------------------------------------------------------===//
// UnitT
//===----------------------------------------------------------------------===//

UnitT UnitT::get(IRContext *context) { return Base::get(context); }

void UnitT::printer(UnitT type, llvm::raw_ostream &os) { os << "unit"; }

std::pair<size_t, size_t> UnitT::calculateSizeAndAlign(UnitT type,
                                                       const StructSizeMap &) {
  return {0, 1};
}

//===----------------------------------------------------------------------===//
// FunctionT
//===----------------------------------------------------------------------===//

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
std::pair<size_t, size_t>
FunctionT::calculateSizeAndAlign(FunctionT type, const StructSizeMap &) {
  return {0, 1};
}

//===----------------------------------------------------------------------===//
// PointerT
//===----------------------------------------------------------------------===//

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

PointerT PointerT::get(IRContext *context, Type pointeeType, bool isConst) {
  return Base::get(context, pointeeType, isConst);
}

Type PointerT::getPointeeType() const { return getImpl()->getPointeeType(); }
bool PointerT::isConst() const { return getImpl()->isConst(); }

void PointerT::printer(PointerT type, llvm::raw_ostream &os) {
  os << type.getPointeeType().toString() << "*"
     << (type.isConst() ? "const" : "");
}

std::pair<size_t, size_t>
PointerT::calculateSizeAndAlign(PointerT type, const StructSizeMap &) {
  auto pointerSize = type.getContext()->getArchitectureBitSize() / BITS_OF_BYTE;
  return {pointerSize, pointerSize};
}

//===----------------------------------------------------------------------===//
// ArrayT
//===----------------------------------------------------------------------===//

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

ArrayT ArrayT::get(IRContext *context, std::size_t size, Type elementType) {
  return Base::get(context, size, elementType);
}

Type ArrayT::getElementType() const { return getImpl()->getElementType(); }
std::size_t ArrayT::getSize() const { return getImpl()->getSize(); }
void ArrayT::printer(ArrayT type, llvm::raw_ostream &os) {
  os << "[" << type.getSize() << " x " << type.getElementType().toString()
     << "]";
}

std::pair<size_t, size_t>
ArrayT::calculateSizeAndAlign(ArrayT type, const StructSizeMap &sizeMap) {
  auto elementType = type.getElementType();
  auto [innerSize, innerAlign] = elementType.getSizeAndAlign(sizeMap);
  auto totalSize = type.getSize() * std::max(innerSize, innerAlign);
  return {totalSize, innerAlign};
}

//===----------------------------------------------------------------------===//
// ConstQualifier
//===----------------------------------------------------------------------===//

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
ConstQualifier ConstQualifier::get(IRContext *context, Type type) {
  return Base::get(context, type);
}

void ConstQualifier::printer(ConstQualifier type, llvm::raw_ostream &os) {
  os << "const " << type.getType().toString();
}

Type ConstQualifier::getType() const { return getImpl()->getType(); }
std::pair<size_t, size_t>
ConstQualifier::calculateSizeAndAlign(ConstQualifier type,
                                      const StructSizeMap &sizeMap) {
  return type.getType().getSizeAndAlign(sizeMap);
}

//===----------------------------------------------------------------------===//
// TupleT
//===----------------------------------------------------------------------===//

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

std::pair<size_t, size_t>
TupleT::calculateSizeAndAlign(TupleT type, const StructSizeMap &sizeMap) {
  auto [totalSize, maxAlign, offsets] =
      getTypeSizeAlignOffsets(type.getElementTypes(), sizeMap);

  return {totalSize, maxAlign};
}

std::tuple<size_t, size_t, llvm::SmallVector<size_t>>
getTypeSizeAlignOffsets(llvm::ArrayRef<Type> fields,
                        const StructSizeMap &sizeMap) {
  if (fields.empty()) {
    return {0, 1, {}};
  }

  llvm::SmallVector<std::pair<size_t, size_t>> fieldsInfo = llvm::map_to_vector(
      fields, [&sizeMap](Type type) { return type.getSizeAndAlign(sizeMap); });

  size_t maxAlign = 0;
  llvm::for_each(fieldsInfo, [&maxAlign](const auto &info) {
    maxAlign = std::max(maxAlign, info.second);
  });

  llvm::SmallVector<size_t> offsets;
  offsets.reserve(fieldsInfo.size());
  size_t currentOffset = 0;
  for (const auto &[fieldSize, fieldAlign] : fieldsInfo) {
    auto pad = currentOffset % fieldAlign
                   ? fieldAlign - (currentOffset % fieldAlign)
                   : 0;
    currentOffset += pad;
    offsets.emplace_back(currentOffset);
    currentOffset += fieldSize;
  }

  auto totalSize = ((currentOffset - 1) / maxAlign + 1) * maxAlign;

  return {totalSize, maxAlign, offsets};
}

void registerBuiltinTypes(IRContext *context) {
  context->registerType<IntT>();
  context->registerType<FloatT>();
  context->registerType<NameStruct>();
  context->registerType<UnitT>();
  context->registerType<FunctionT>();
  context->registerType<PointerT>();
  context->registerType<ConstQualifier>();
  context->registerType<TupleT>();
  context->registerType<ArrayT>();
}

} // namespace kecc::ir
