#ifndef KECC_C_PARSE_AST_H
#define KECC_C_PARSE_AST_H

#include "kecc/c/Diag.h"
#include "kecc/c/Visitor.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Frontend/ASTUnit.h"

#include "kecc/c/Clang.h"

namespace kecc::c {

class ParseAST {
public:
  ParseAST(llvm::StringRef source, llvm::StringRef filename)
      : source(source), filename(filename) {}

  void parse();

  clang::ASTUnit *getASTUnit() const { return astUnit.get(); }

  void dump() const;

private:
  llvm::StringRef source;
  llvm::StringRef filename;
  std::unique_ptr<clang::ASTUnit> astUnit;
};

class AssertImpl : public TypeVisitor<AssertImpl>,
                   public StmtVisitor<AssertImpl>,
                   public DeclVisitor<AssertImpl> {
public:
  AssertImpl(ClangDiagManager *diags, const clang::LangOptions &opt);

  void VisitBinaryOperator(const BinaryOperator *expr);
  void VisitUnaryOperator(const UnaryOperator *expr);
  void VisitUnaryExprOrTypeTraitExpr(const UnaryExprOrTypeTraitExpr *expr);
  void VisitIntegerLiteral(const IntegerLiteral *expr);
  void VisitFloatingLiteral(const FloatingLiteral *expr);
  void VisitParenExpr(const ParenExpr *expr);
  void VisitImplicitCastExpr(const ImplicitCastExpr *expr);
  void VisitExplicitCastExpr(const ExplicitCastExpr *expr);
  void VisitInitListExpr(const InitListExpr *expr);
  void VisitMemberExpr(const MemberExpr *expr);
  void VisitConditionalOperator(const ConditionalOperator *expr);

  void VisitVarDecl(const VarDecl *decl);
  void VisitValueDecl(const ValueDecl *decl);
  void VisitFunctionDecl(const FunctionDecl *decl);
  void VisitFieldDecl(const FieldDecl *decl);
  void VisitRecordDecl(const RecordDecl *decl);
  void VisitTypedefDecl(const TypedefDecl *decl);
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

  void VisitQualType(QualType qt, const SourceLocation &loc);
  void VisitArrayType(const ArrayType *T, const SourceLocation &loc);
  void VisitRecordType(const RecordType *T, const SourceLocation &loc);
  void VisitTypedefType(const TypedefType *T, const SourceLocation &loc);
  void VisitPointerType(const PointerType *T, const SourceLocation &loc);
  void VisitFunctionType(const FunctionType *T, const SourceLocation &loc);
  void VisitBuiltinType(const BuiltinType *T, const SourceLocation &loc);

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
