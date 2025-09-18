#ifndef KECC_C_AST_VISITOR_H
#define KECC_C_AST_VISITOR_H

#include "kecc/c/Clang.h"
#include "kecc/c/Diag.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/Support/ErrorHandling.h"

namespace kecc::c {

template <template <typename, typename> typename ConcreteVisitor>
class VisitorBase {
public:
  VisitorBase(ClangDiagManager *diag) : diag(diag) {}

protected:
  DiagnosticBuilder report(const SourceLocation &loc,
                           ClangDiagManager::DiagID diagID) {
    return diag->report(loc, diagID);
  }

private:
  ClangDiagManager *diag;
};

template <typename ImplClass, typename RetTy = void>
class TypeVisitor : public VisitorBase<TypeVisitor> {
public:
  TypeVisitor(ClangDiagManager *diag) : VisitorBase(diag) {}

#define DISPATCH(CLASS)                                                        \
  return static_cast<ImplClass *>(this)->Visit##CLASS(                         \
      static_cast<const CLASS *>(T), S)

  RetTy Visit(const Type *T, const SourceLocation &S) {
    switch (T->getTypeClass()) {
#define ABSTRACT_TYPE(CLASS, PARENT)
#define TYPE(CLASS, PARENT)                                                    \
  case Type::CLASS:                                                            \
    DISPATCH(CLASS##Type);
#include "clang/AST/TypeNodes.inc"
    }
    llvm_unreachable("Unknown type class");
  }

#define TYPE(CLASS, PARENT)                                                    \
  RetTy Visit##CLASS##Type(const CLASS##Type *T, const SourceLocation &S) {    \
    DISPATCH(PARENT);                                                          \
  }
#include "clang/AST/TypeNodes.inc"

  RetTy VisitType(const Type *T, const SourceLocation &S) {
    report(S, ClangDiagManager::unsupported_type) << T->getTypeClassName();
    return RetTy();
  }

#undef DISPATCH
};

template <typename ImplClass, typename RetTy = void>
class StmtVisitor : public VisitorBase<StmtVisitor> {
public:
  StmtVisitor(ClangDiagManager *diag) : VisitorBase(diag) {}

#define DISPATCH(NAME, CLASS)                                                  \
  return static_cast<ImplClass *>(this)->Visit##NAME(                          \
      static_cast<const CLASS *>(S))

  RetTy Visit(const Stmt *S) {

    switch (S->getStmtClass()) {
#define ABSTRACT_STMT(CLASS)
#define STMT(CLASS, PARENT)                                                    \
  case Stmt::CLASS##Class:                                                     \
    DISPATCH(CLASS, CLASS);
#include "clang/AST/StmtNodes.inc"
    default:
      llvm_unreachable("Unknown stmt kind");
    }
  }

#define STMT(CLASS, PARENT)                                                    \
  RetTy Visit##CLASS(const CLASS *S) { DISPATCH(PARENT, PARENT); }
#include "clang/AST/StmtNodes.inc"

  RetTy VisitStmt(const Stmt *S) {
    report(S->getBeginLoc(), ClangDiagManager::unsupported_stmt)
        << S->getStmtClassName();
    return RetTy();
  }

#undef DISPATCH
};

template <typename ImplClass, typename RetTy = void>
class DeclVisitor : public VisitorBase<DeclVisitor> {
public:
  DeclVisitor(ClangDiagManager *diag) : VisitorBase(diag) {}

#define DISPATCH(NAME, CLASS)                                                  \
  return static_cast<ImplClass *>(this)->Visit##NAME(                          \
      static_cast<const CLASS *>(D))

  RetTy Visit(const Decl *D) {
    switch (D->getKind()) {
#define ABSTRACT_DECL(CLASS)
#define DECL(CLASS, PARENT)                                                    \
  case Decl::CLASS:                                                            \
    DISPATCH(CLASS##Decl, CLASS##Decl);
#include "clang/AST/DeclNodes.inc"
    }
    llvm_unreachable("Unknown decl kind");
  }

#define DECL(CLASS, PARENT)                                                    \
  RetTy Visit##CLASS##Decl(const CLASS##Decl *D) { DISPATCH(PARENT, PARENT); }
#include "clang/AST/DeclNodes.inc"

  RetTy VisitDecl(const Decl *D) {
    report(D->getBeginLoc(), ClangDiagManager::unsupported_decl)
        << D->getDeclKindName();
    return RetTy();
  }

#undef DISPATCH
};

} // namespace kecc::c

#endif // KECC_C_AST_VISITOR_H
