#include "kecc/c/ParseAST.h"
#include "kecc/c/Diag.h"
#include "kecc/utils/LogicalResult.h"
#include "clang/AST/Decl.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/TypeTraits.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Tooling/Tooling.h"

namespace kecc::c {

utils::LogicalResult ParseAST::parse() {
  std::vector<std::string> args{"--target=riscv64-unknown-linux-gnu"};
  if (ignoreWarnings) {
    args.push_back("-w");
  }

  auto astUnit =
      clang::tooling::buildASTFromCodeWithArgs(source, args, filename);
  if (astUnit->getDiagnostics().hasErrorOccurred()) {
    return utils::LogicalResult::error();
  }

  astUnit->getDiagnostics().setClient(new clang::TextDiagnosticPrinter(
      *diagOS, &astUnit->getDiagnostics().getDiagnosticOptions()));
  ClangDiagManager diagManager(&astUnit->getDiagnostics(), astUnit.get());
  AssertImpl assertImpl(&diagManager, astUnit->getLangOpts());
  assertImpl.Visit(astUnit->getASTContext().getTranslationUnitDecl());

  if (diagManager.hasError()) {
    return utils::LogicalResult::error();
  }

  this->astUnit = std::move(astUnit);
  return utils::LogicalResult::success();
}

void ParseAST::dump() const {
  if (astUnit) {
    astUnit->getASTContext().getTranslationUnitDecl()->dump();
  }
}

AssertImpl::AssertImpl(ClangDiagManager *diag, const clang::LangOptions &opt)
    : diag(diag), opt(opt), TypeVisitor(diag), StmtVisitor(diag),
      DeclVisitor(diag) {}

void AssertImpl::Visit(const Type *type, const SourceLocation &loc) {
  TypeVisitor::Visit(type, loc);
}

void AssertImpl::Visit(const Stmt *stmt) { StmtVisitor::Visit(stmt); }

void AssertImpl::Visit(const Decl *decl) { DeclVisitor::Visit(decl); }

void AssertImpl::VisitBinaryOperator(const BinaryOperator *expr) {
  auto opKind = expr->getOpcode();

  switch (opKind) {
  case clang::BO_PtrMemD:
  case clang::BO_PtrMemI:
  case clang::BO_Cmp:
    report(expr->getOperatorLoc(), DiagID::unsupported_operator)
        << expr->getOpcodeStr();
    break;
  case clang::BO_Mul:
  case clang::BO_Div:
  case clang::BO_Rem:
  case clang::BO_Add:
  case clang::BO_Sub:
  case clang::BO_Shl:
  case clang::BO_Shr:
  case clang::BO_LT:
  case clang::BO_GT:
  case clang::BO_LE:
  case clang::BO_GE:
  case clang::BO_EQ:
  case clang::BO_NE:
  case clang::BO_And:
  case clang::BO_Xor:
  case clang::BO_Or:
  case clang::BO_LAnd:
  case clang::BO_LOr:
  case clang::BO_Assign:
  case clang::BO_MulAssign:
  case clang::BO_DivAssign:
  case clang::BO_RemAssign:
  case clang::BO_AddAssign:
  case clang::BO_SubAssign:
  case clang::BO_ShlAssign:
  case clang::BO_ShrAssign:
  case clang::BO_AndAssign:
  case clang::BO_XorAssign:
  case clang::BO_OrAssign:
  case clang::BO_Comma:
    break;
  }
}

void AssertImpl::VisitUnaryOperator(const UnaryOperator *expr) {
  auto opKind = expr->getOpcode();

  switch (opKind) {
  case clang::UO_PostInc:
  case clang::UO_PostDec:
  case clang::UO_PreInc:
  case clang::UO_PreDec:
  case clang::UO_AddrOf:
  case clang::UO_Deref:
  case clang::UO_Plus:
  case clang::UO_Minus:
  case clang::UO_Not:
  case clang::UO_LNot:
    break;
  default:
    report(expr->getOperatorLoc(), DiagID::unsupported_operator)
        << expr->getOpcodeStr(expr->getOpcode());
    break;
  }
}

void AssertImpl::VisitUnaryExprOrTypeTraitExpr(
    const UnaryExprOrTypeTraitExpr *expr) {
  switch (expr->getKind()) {
  case clang::UETT_DataSizeOf:
  case clang::UETT_VecStep:
  case clang::UETT_PreferredAlignOf:
  case clang::UETT_OpenMPRequiredSimdAlign:
  case clang::UETT_VectorElements:
    report(expr->getOperatorLoc(), DiagID::unsupported_expr)
        << clang::getTraitSpelling(expr->getKind());
    break;
  case clang::UETT_SizeOf:
  case clang::UETT_AlignOf:
    break;
  }
}

void AssertImpl::VisitIntegerLiteral(const IntegerLiteral *expr) {}
void AssertImpl::VisitCharacterLiteral(const CharacterLiteral *expr) {}
void AssertImpl::VisitFloatingLiteral(const FloatingLiteral *expr) {}
void AssertImpl::VisitParenExpr(const ParenExpr *expr) {
  StmtVisitor::Visit(expr->getSubExpr());
}
void AssertImpl::VisitArraySubscriptExpr(const ArraySubscriptExpr *expr) {
  StmtVisitor::Visit(expr->getBase());
  StmtVisitor::Visit(expr->getIdx());
}

void AssertImpl::VisitCastExpr(const CastExpr *expr) {
  switch (expr->getCastKind()) {
  case clang::CK_LValueToRValue:
  case clang::CK_IntegralCast:
  case clang::CK_IntegralToBoolean:
  case clang::CK_IntegralToFloating:
  case clang::CK_BitCast:
  case clang::CK_NoOp:
  case clang::CK_ArrayToPointerDecay:
  case clang::CK_FunctionToPointerDecay:
  case clang::CK_NullToPointer:
  case clang::CK_IntegralToPointer:
  case clang::CK_PointerToIntegral:
  case clang::CK_PointerToBoolean:
  case clang::CK_ToVoid:
  case clang::CK_FloatingToIntegral:
  case clang::CK_FloatingToBoolean:
  case clang::CK_BooleanToSignedIntegral:
  case clang::CK_FloatingCast:
    break;
  default:
    report(expr->getExprLoc(), DiagID::unsupported_cast_kind)
        << clang::CastExpr::getCastKindName(expr->getCastKind());
    break;
  }

  // c doesn't support cast path
  assert(expr->path_empty() && "cast path is not supported");

  if (expr->hasStoredFPFeatures())
    report(expr->getExprLoc(), DiagID::unsupported_cast_fp_features);
}
void AssertImpl::VisitImplicitCastExpr(const clang::ImplicitCastExpr *expr) {
  StmtVisitor::Visit(expr->getSubExpr());
  VisitQualType(expr->getType(), expr->getExprLoc());
  VisitCastExpr(expr);
}
void AssertImpl::VisitCStyleCastExpr(const CStyleCastExpr *expr) {
  StmtVisitor::Visit(expr->getSubExpr());
  VisitQualType(expr->getType(), expr->getExprLoc());
  VisitCastExpr(expr);
}

void AssertImpl::VisitInitListExpr(const InitListExpr *expr) {
  for (const auto *init : expr->inits()) {
    StmtVisitor::Visit(init);
  }
}
void AssertImpl::VisitMemberExpr(const MemberExpr *expr) {
  StmtVisitor::Visit(expr->getBase());
  if (auto *decl = expr->getMemberDecl()) {
    DeclVisitor<AssertImpl>::Visit(decl);
  }
}
void AssertImpl::VisitConditionalOperator(const ConditionalOperator *expr) {
  StmtVisitor::Visit(expr->getCond());
  StmtVisitor::Visit(expr->getTrueExpr());
  StmtVisitor::Visit(expr->getFalseExpr());
}
void AssertImpl::VisitCallExpr(const CallExpr *expr) {
  StmtVisitor::Visit(expr->getCallee());
  for (const auto *arg : expr->arguments()) {
    StmtVisitor::Visit(arg);
  }
}
void AssertImpl::VisitDeclRefExpr(const DeclRefExpr *expr) {
  if (expr->hasQualifier()) {
    report(expr->getExprLoc(), DiagID::unsupported_qualifier)
        << "nested name specifier";
  }

  if (expr->hasTemplateKeyword()) {
    report(expr->getExprLoc(), DiagID::unsupported_qualifier)
        << "template keyword";
  }

  if (expr->hasExplicitTemplateArgs()) {
    report(expr->getExprLoc(), DiagID::unsupported_qualifier)
        << "explicit template arguments";
  }

  auto name = expr->getNameInfo().getName();
  if (!name.isIdentifier()) {
    report(expr->getExprLoc(), DiagID::unsupported_expr)
        << "non-identifier name";
  }
}

void AssertImpl::VisitTranslationUnitDecl(const TranslationUnitDecl *decl) {
  for (auto I = decl->decls_begin(), E = decl->decls_end(); I != E; ++I) {
    if (const auto *typedefDecl = llvm::dyn_cast<TypedefDecl>(*I)) {
      if (typedefDecl->isImplicit())
        continue;
    }
    DeclVisitor::Visit(*I);
  }
}

void AssertImpl::VisitVarDecl(const VarDecl *decl) {
  VisitQualType(decl->getType(), decl->getTypeSpecStartLoc());
  if (decl->hasInit()) {
    StmtVisitor::Visit(decl->getInit());
  }
}

void AssertImpl::VisitFunctionDecl(const FunctionDecl *decl) {
  VisitQualType(decl->getType(), decl->getLocation());
  for (const auto *param : decl->parameters()) {
    DeclVisitor::Visit(param);
  }
  if (decl->isVariadic())
    report(decl->getFunctionTypeLoc().getRParenLoc(), DiagID::variadic_param);

  if (decl->hasBody())
    StmtVisitor::Visit(decl->getBody());
}

void AssertImpl::VisitFieldDecl(const FieldDecl *decl) {
  VisitQualType(decl->getType(), decl->getLocation());
  if (decl->isBitField()) {
    report(decl->getLocation(), DiagID::bit_field);
  }
}

void AssertImpl::VisitRecordDecl(const RecordDecl *decl) {
  switch (decl->getTagKind()) {
  case clang::TagTypeKind::Interface:
  case clang::TagTypeKind::Union:
  case clang::TagTypeKind::Class:
  case clang::TagTypeKind::Enum:
    report(decl->getLocation(), DiagID::unsupported_decl)
        << decl->getKindName();
    return;
  case clang::TagTypeKind::Struct:
    break;
  };

  for (const auto *field : decl->fields()) {
    DeclVisitor::Visit(field);
  }
}

void AssertImpl::VisitTypedefDecl(const TypedefDecl *decl) {
  VisitQualType(decl->getUnderlyingType(), decl->getLocation());
}

void AssertImpl::VisitDeclStmt(const DeclStmt *stmt) {
  for (const auto *decl : stmt->decls()) {
    DeclVisitor::Visit(decl);
  }
}
void AssertImpl::VisitBreakStmt(const BreakStmt *stmt) {}
void AssertImpl::VisitContinueStmt(const ContinueStmt *stmt) {}
void AssertImpl::VisitCompoundStmt(const CompoundStmt *stmt) {
  for (const auto *s : stmt->body()) {
    StmtVisitor::Visit(s);
  }
}

void AssertImpl::VisitDoStmt(const DoStmt *stmt) {
  StmtVisitor::Visit(stmt->getBody());
  StmtVisitor::Visit(stmt->getCond());
}
void AssertImpl::VisitForStmt(const ForStmt *stmt) {
  if (stmt->getInit()) {
    StmtVisitor::Visit(stmt->getInit());
  }
  if (stmt->getCond()) {
    StmtVisitor::Visit(stmt->getCond());
  }
  if (stmt->getInc()) {
    StmtVisitor::Visit(stmt->getInc());
  }
  StmtVisitor::Visit(stmt->getBody());
}

void AssertImpl::VisitWhileStmt(const WhileStmt *stmt) {
  StmtVisitor::Visit(stmt->getCond());
  StmtVisitor::Visit(stmt->getBody());
}

void AssertImpl::VisitSwitchStmt(const SwitchStmt *stmt) {
  if (stmt->getInit()) {
    report(stmt->getInit()->getBeginLoc(), DiagID::switch_init);
  }
  StmtVisitor::Visit(stmt->getCond());
  StmtVisitor::Visit(stmt->getBody());
}

void AssertImpl::VisitSwitchCase(const SwitchCase *stmt) {
  StmtVisitor::Visit(stmt->getSubStmt());
}

void AssertImpl::VisitCaseStmt(const CaseStmt *stmt) {
  StmtVisitor::Visit(stmt->getLHS());
  if (stmt->getRHS()) {
    report(stmt->getRHS()->getBeginLoc(), DiagID::switch_range);
  }
  StmtVisitor::Visit(stmt->getSubStmt());
}

void AssertImpl::VisitDefaultStmt(const DefaultStmt *stmt) {
  StmtVisitor::Visit(stmt->getSubStmt());
}

void AssertImpl::VisitIfStmt(const IfStmt *stmt) {
  if (stmt->getInit()) {
    report(stmt->getInit()->getBeginLoc(), DiagID::if_init);
  }
  StmtVisitor::Visit(stmt->getCond());
  StmtVisitor::Visit(stmt->getThen());
  if (stmt->getElse()) {
    StmtVisitor::Visit(stmt->getElse());
  }
}

void AssertImpl::VisitReturnStmt(const ReturnStmt *stmt) {
  if (stmt->getRetValue()) {
    StmtVisitor::Visit(stmt->getRetValue());
  }
}
void AssertImpl::VisitNullStmt(const NullStmt *stmt) {}

void AssertImpl::VisitQualType(QualType qt, const SourceLocation &loc) {
  if (qt.hasQualifiers()) {
    if (qt.isVolatileQualified())
      report(loc, DiagID::unsupported_qualifier) << "volatile";

    if (qt.isRestrictQualified())
      report(loc, DiagID::unsupported_qualifier) << "restrict";
  }

  TypeVisitor::Visit(qt.getTypePtr(), loc);
}

void AssertImpl::VisitArrayType(const ArrayType *T, const SourceLocation &loc) {
  if (T->isIncompleteArrayType())
    report(loc, DiagID::unsupported_type) << "incomplete array type";
  else if (T->isDependentSizedArrayType())
    report(loc, DiagID::unsupported_type) << "dependent sized array type";
  else if (T->isVariableArrayType())
    report(loc, DiagID::unsupported_type) << "variable array type";

  switch (T->getSizeModifier()) {
  case clang::ArraySizeModifier::Static:
    report(loc, DiagID::unsupported_qualifier) << "array size modifier(static)";
    break;
  case clang::ArraySizeModifier::Star:
    report(loc, DiagID::unsupported_qualifier) << "array size modifier(*)";
    break;
  case clang::ArraySizeModifier::Normal:
    break;
  }

  VisitQualType(T->getElementType(), loc);
}

void AssertImpl::VisitRecordType(const RecordType *T,
                                 const SourceLocation &loc) {
  // Do nothing
}
void AssertImpl::VisitElaboratedType(const ElaboratedType *T,
                                     const SourceLocation &loc) {
  VisitQualType(T->getNamedType(), loc);
}

void AssertImpl::VisitTypedefType(const TypedefType *T,
                                  const SourceLocation &loc) {
  // Do nothing
}

void AssertImpl::VisitPointerType(const clang::PointerType *T,
                                  const SourceLocation &loc) {
  VisitQualType(T->getPointeeType(), loc);
}

void AssertImpl::VisitFunctionType(const clang::FunctionType *T,
                                   const SourceLocation &loc) {
  VisitQualType(T->getReturnType(), loc);

  if (const FunctionProtoType *proto = T->getAs<FunctionProtoType>()) {
    for (unsigned idx = 0; idx < proto->getNumParams(); ++idx) {
      auto EPI = proto->getExtParameterInfo(idx);
      if (EPI.getOpaqueValue() != (unsigned char)0)
        report(loc, DiagID::ext_param_info);

      VisitQualType(proto->getParamType(idx), loc);
    }

    if (proto->isVariadic())
      report(loc, DiagID::variadic_param);
  }
}

void AssertImpl::VisitBuiltinType(const BuiltinType *T,
                                  const SourceLocation &loc) {
  switch (T->getKind()) {
  case clang::BuiltinType::Void:
  case clang::BuiltinType::Bool:
  case clang::BuiltinType::Char_U:
  case clang::BuiltinType::UChar:
  case clang::BuiltinType::UShort:
  case clang::BuiltinType::UInt:
  case clang::BuiltinType::ULong:
  case clang::BuiltinType::ULongLong:
  case clang::BuiltinType::SChar:
  case clang::BuiltinType::Short:
  case clang::BuiltinType::Int:
  case clang::BuiltinType::Long:
  case clang::BuiltinType::LongLong:
  case clang::BuiltinType::Float:
  case clang::BuiltinType::Double:
    break;
  default: {
    clang::PrintingPolicy policy(opt);
    report(loc, DiagID::unsupported_type) << T->getName(policy);
  }
  }
}

void AssertImpl::VisitParenType(const clang::ParenType *T,
                                const SourceLocation &loc) {
  VisitQualType(T->getInnerType(), loc);
}

} // namespace kecc::c
