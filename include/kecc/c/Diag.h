#ifndef KECC_C_DIAG_H
#define KECC_C_DIAG_H

#include "kecc/c/Clang.h"
#include "clang/AST/ASTContext.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Frontend/ASTUnit.h"
#include "llvm/ADT/DenseMap.h"

namespace kecc::c {

class ClangDiagManager {
public:
  enum DiagID {
    unsupported_operator = 0,
    unsupported_expr,
    unsupported_stmt,
    unsupported_decl,
    unsupported_type,
    switch_init,
    switch_range,
    if_init,
    bit_field,
    unsupported_qualifier,
    ext_param_info,
    variadic_param,
    unsupported_cast_kind,
    unsupported_cast_fp_features,
    DIAG_ID_COUNT
  };

  ClangDiagManager(DiagnosticsEngine *diags, clang::ASTUnit *ast)
      : diags(diags) {
    diags->getClient()->BeginSourceFile(ast->getLangOpts(),
                                        &ast->getPreprocessor());
    // clang-format off
    diagToClangDiagMap[unsupported_operator]      = diags->getCustomDiagID(DiagnosticsEngine::Error, "unsupported operator(%0)");
    diagToClangDiagMap[unsupported_expr]          = diags->getCustomDiagID(DiagnosticsEngine::Error, "unsupported expr : %0");
    diagToClangDiagMap[unsupported_stmt]          = diags->getCustomDiagID(DiagnosticsEngine::Error, "unsupported stmt : %0");
    diagToClangDiagMap[unsupported_decl]          = diags->getCustomDiagID(DiagnosticsEngine::Error, "unsupported decl : %0");
    diagToClangDiagMap[unsupported_type]          = diags->getCustomDiagID(DiagnosticsEngine::Error, "unsupported type : %0");
    diagToClangDiagMap[switch_init]               = diags->getCustomDiagID(DiagnosticsEngine::Error, "switch condition cannot be an init-statement");
    diagToClangDiagMap[switch_range]              = diags->getCustomDiagID(DiagnosticsEngine::Error, "switch case cannot be a range");
    diagToClangDiagMap[if_init]                   = diags->getCustomDiagID(DiagnosticsEngine::Error, "if condition cannot be an init-statement");
    diagToClangDiagMap[bit_field]                 = diags->getCustomDiagID(DiagnosticsEngine::Error, "bit-field is not supported");
    diagToClangDiagMap[unsupported_qualifier]     = diags->getCustomDiagID(DiagnosticsEngine::Error, "unsupported qualifier : %0");
    diagToClangDiagMap[ext_param_info]            = diags->getCustomDiagID(DiagnosticsEngine::Error, "additional parameter information");
    diagToClangDiagMap[variadic_param]            = diags->getCustomDiagID(DiagnosticsEngine::Error, "variadic arguments is not supported");
    diagToClangDiagMap[unsupported_cast_kind]     = diags->getCustomDiagID(DiagnosticsEngine::Error, "unsupported cast kind : %0");
    diagToClangDiagMap[unsupported_cast_fp_features]  = diags->getCustomDiagID(DiagnosticsEngine::Error, "casting with floating-point environment is not supported");
    // clang-format on
  }

  ~ClangDiagManager() { diags->getClient()->EndSourceFile(); }

  DiagnosticBuilder report(SourceLocation loc, DiagID id) {
    assert(id < DIAG_ID_COUNT);
    return diags->Report(loc, diagToClangDiagMap[id]);
  }

  bool hasError() const { return diags->hasErrorOccurred(); }

private:
  unsigned diagToClangDiagMap[DIAG_ID_COUNT];
  DiagnosticsEngine *diags;
};

} // namespace kecc::c

#endif // KECC_C_DIAG_H
