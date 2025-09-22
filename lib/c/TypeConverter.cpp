#include "kecc/c/TypeConverter.h"
#include "kecc/c/IRGenerator.h"
#include "kecc/ir/IRTypes.h"
#include "clang/AST/Type.h"

namespace kecc::c {

ir::Type TypeConverter::VisitQualType(QualType qt, const SourceLocation &loc) {
  auto T = TypeVisitor::Visit(qt.getTypePtr(), loc);
  assert(T && "Type conversion failed");
  if (qt.isConstQualified())
    T = ir::ConstQualifier::get(ctx, T);
  return T;
}

ir::Type TypeConverter::VisitArrayType(const ArrayType *T,
                                       const SourceLocation &loc) {
  auto elemType = VisitQualType(T->getElementType(), loc);
  assert(elemType && "Element type conversion failed");
  auto *constArray = llvm::dyn_cast<ConstantArrayType>(T);
  assert(constArray && "Only constant array is supported");

  auto arraySize = constArray->getSize().getZExtValue();
  return ir::ArrayT::get(ctx, arraySize, elemType);
}

ir::Type TypeConverter::VisitRecordType(const RecordType *T,
                                        const SourceLocation &loc) {
  auto decl = T->getDecl();
  assert(decl && "RecordType without declaration");
  auto name = decl->getName();
  if (name.empty())
    name = recordDeclMgr->getRecordDeclID(decl, *this);

  return ir::NameStruct::get(ctx, name);
}

ir::Type TypeConverter::VisitTypedefType(const TypedefType *T,
                                         const SourceLocation &loc) {
  auto decl = T->getDecl();
  assert(decl && "TypedefType without declaration");
  auto underlyingType = decl->getUnderlyingType();
  return VisitQualType(underlyingType, loc);
}

ir::Type TypeConverter::VisitPointerType(const PointerType *T,
                                         const SourceLocation &loc) {
  auto pointeeType = VisitQualType(T->getPointeeType(), loc);
  assert(pointeeType && "Pointee type conversion failed");
  if (auto constQ = pointeeType.dyn_cast<ir::ConstQualifier>()) {
    return ir::PointerT::get(ctx, constQ.getType(), true);
  }
  return ir::PointerT::get(ctx, pointeeType);
}

ir::Type TypeConverter::VisitFunctionType(const FunctionType *T,
                                          const SourceLocation &loc) {
  auto retType = VisitQualType(T->getReturnType(), loc);
  assert(retType && "Return type conversion failed");
  std::vector<ir::Type> paramTypes;
  if (auto funcProto = llvm::dyn_cast<FunctionProtoType>(T)) {
    for (const auto &paramType : funcProto->param_types()) {
      auto paramTy = VisitQualType(paramType, loc);
      assert(paramTy && "Parameter type conversion failed");
      paramTypes.emplace_back(paramTy);
    }
  }
  return ir::FunctionT::get(ctx, retType, paramTypes);
}

ir::Type TypeConverter::VisitBuiltinType(const clang::BuiltinType *T,
                                         const SourceLocation &loc) {
  switch (T->getKind()) {
  case BuiltinType::Void:
    return ir::UnitT::get(ctx);
  case BuiltinType::Bool:
    return ir::IntT::get(ctx, 1, false);
  case BuiltinType::Char_U:
    return ir::IntT::get(ctx, 8, false);
  case BuiltinType::UChar:
    return ir::IntT::get(ctx, 8, false);
  case BuiltinType::UShort:
    return ir::IntT::get(ctx, 16, false);
  case BuiltinType::UInt:
    return ir::IntT::get(ctx, 32, false);
  case BuiltinType::ULong:
    return ir::IntT::get(ctx, 32, false);
  case BuiltinType::ULongLong:
    return ir::IntT::get(ctx, 64, false);
  case BuiltinType::SChar:
    return ir::IntT::get(ctx, 8, true);
  case BuiltinType::Short:
    return ir::IntT::get(ctx, 16, true);
  case BuiltinType::Int:
    return ir::IntT::get(ctx, 32, true);
  case BuiltinType::Long:
    return ir::IntT::get(ctx, 32, true);
  case BuiltinType::LongLong:
    return ir::IntT::get(ctx, 64, true);
  case BuiltinType::Float:
    return ir::FloatT::get(ctx, 32);
  case BuiltinType::Double:
    return ir::FloatT::get(ctx, 64);
  default:
    llvm_unreachable("Unsupported builtin type");
  }
}

ir::Type TypeConverter::VisitParenType(const clang::ParenType *T,
                                       const SourceLocation &loc) {
  return VisitQualType(T->getInnerType(), loc);
}

ir::Type TypeConverter::VisitElaboratedType(const ElaboratedType *T,
                                            const SourceLocation &loc) {
  return VisitQualType(T->getNamedType(), loc);
}

} // namespace kecc::c
