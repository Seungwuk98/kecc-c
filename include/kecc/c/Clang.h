#ifndef KECC_C_CLANG_H
#define KECC_C_CLANG_H

#include "clang/AST/AST.h"
#include "clang/AST/DeclFriend.h"
#include "clang/AST/DeclOpenMP.h"

namespace kecc::c {
using clang::DiagnosticBuilder;
using clang::DiagnosticIDs;
using clang::DiagnosticsEngine;
using clang::SourceLocation;
using clang::SourceRange;

using clang::Attr;
using clang::Decl;
using clang::QualType;
using clang::Stmt;
using clang::Type;

#define ATTR(NAME) using clang::NAME##Attr;
#include "clang/Basic/AttrList.inc"

#define ABSTRACT_DECL(CLASS) CLASS
#define DECL(CLASS, PARENT) using clang::CLASS##Decl;
#include "clang/AST/DeclNodes.inc"

#define ABSTRACT_STMT(CLASS) CLASS
#define STMT(CLASS, PARENT) using clang::CLASS;
#include "clang/AST/StmtNodes.inc"

#define ABSTRACT_TYPE(CLASS, PARENT) using clang::CLASS##Type;
#define TYPE(CLASS, PARENT) using clang::CLASS##Type;
#include "clang/AST/TypeNodes.inc"

} // namespace kecc::c

#endif // KECC_C_CLANG_H
