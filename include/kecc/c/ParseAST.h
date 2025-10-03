#ifndef KECC_C_PARSE_AST_H
#define KECC_C_PARSE_AST_H

#include "kecc/c/Clang.h"
#include "kecc/c/Diag.h"
#include "kecc/c/Visitor.h"
#include "kecc/utils/LogicalResult.h"
#include "clang/AST/Expr.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Frontend/ASTUnit.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"

namespace kecc::c {

class ParseAST {
public:
  ParseAST(llvm::StringRef source, llvm::StringRef filename,
           llvm::raw_ostream &diagOS = llvm::errs(),
           bool ignoreWarnings = false)
      : source(source), filename(filename), ignoreWarnings(ignoreWarnings),
        diagOS(&diagOS) {}

  utils::LogicalResult parse();

  clang::ASTUnit *getASTUnit() const { return astUnit.get(); }
  std::unique_ptr<clang::ASTUnit> releaseASTUnit() {
    return std::move(astUnit);
  }

  void dump() const;

private:
  llvm::StringRef source;
  llvm::StringRef filename;
  std::unique_ptr<clang::ASTUnit> astUnit;
  llvm::raw_ostream *diagOS;
  bool ignoreWarnings;
};

class AssertImpl : public TypeVisitor<AssertImpl>,
                   public StmtVisitor<AssertImpl>,
                   public DeclVisitor<AssertImpl> {
public:
  AssertImpl(ClangDiagManager *diags, const clang::LangOptions &opt);

  void Visit(const Type *type, const SourceLocation &loc);
  void Visit(const Stmt *stmt);
  void Visit(const Decl *decl);

  void VisitBinaryOperator(const BinaryOperator *expr);
  void VisitUnaryOperator(const UnaryOperator *expr);
  void VisitUnaryExprOrTypeTraitExpr(const UnaryExprOrTypeTraitExpr *expr);
  void VisitIntegerLiteral(const IntegerLiteral *expr);
  void VisitFloatingLiteral(const FloatingLiteral *expr);
  void VisitCharacterLiteral(const CharacterLiteral *expr);
  void VisitParenExpr(const ParenExpr *expr);
  void VisitArraySubscriptExpr(const ArraySubscriptExpr *expr);
  void VisitConstantExpr(const ConstantExpr *expr);

  void VisitCastExpr(const CastExpr *expr);
  void VisitImplicitCastExpr(const ImplicitCastExpr *expr);
  void VisitCStyleCastExpr(const CStyleCastExpr *expr);

  void VisitInitListExpr(const InitListExpr *expr);
  void VisitMemberExpr(const MemberExpr *expr);
  void VisitConditionalOperator(const ConditionalOperator *expr);
  void VisitCallExpr(const CallExpr *expr);
  void VisitDeclRefExpr(const DeclRefExpr *expr);

  void VisitTranslationUnitDecl(const TranslationUnitDecl *decl);
  void VisitVarDecl(const VarDecl *decl);
  void VisitFunctionDecl(const FunctionDecl *decl);
  void VisitFieldDecl(const FieldDecl *decl);
  void VisitRecordDecl(const RecordDecl *decl);
  void VisitTypedefDecl(const TypedefDecl *decl);

  void VisitDeclStmt(const DeclStmt *stmt);
  void VisitBreakStmt(const BreakStmt *stmt);
  void VisitContinueStmt(const ContinueStmt *stmt);
  void VisitCompoundStmt(const CompoundStmt *stmt);
  void VisitDoStmt(const DoStmt *stmt);
  void VisitForStmt(const ForStmt *stmt);
  void VisitWhileStmt(const WhileStmt *stmt);
  void VisitSwitchStmt(const SwitchStmt *stmt);
  void VisitSwitchCase(const SwitchCase *stmt);
  void VisitCaseStmt(const CaseStmt *stmt);
  void VisitDefaultStmt(const DefaultStmt *stmt);
  void VisitIfStmt(const IfStmt *stmt);
  void VisitReturnStmt(const ReturnStmt *stmt);
  void VisitNullStmt(const NullStmt *stmt);

  void VisitQualType(QualType qt, const SourceLocation &loc);
  void VisitArrayType(const ArrayType *T, const SourceLocation &loc);
  void VisitRecordType(const RecordType *T, const SourceLocation &loc);
  void VisitElaboratedType(const clang::ElaboratedType *T,
                           const SourceLocation &loc);
  void VisitTypedefType(const TypedefType *T, const SourceLocation &loc);
  void VisitPointerType(const PointerType *T, const SourceLocation &loc);
  void VisitFunctionType(const FunctionType *T, const SourceLocation &loc);
  void VisitDecayedType(const clang::DecayedType *T, const SourceLocation &loc);
  void VisitBuiltinType(const BuiltinType *T, const SourceLocation &loc);
  void VisitParenType(const ParenType *T, const SourceLocation &loc);

private:
  using DiagID = ClangDiagManager::DiagID;

  DiagnosticBuilder report(const SourceLocation &loc,
                           ClangDiagManager::DiagID diagID) {
    return diag->report(loc, diagID);
  }

  ClangDiagManager *diag;
  const clang::LangOptions &opt;
};

} // namespace kecc::c

#endif // KECC_C_PARSE_AST_H
