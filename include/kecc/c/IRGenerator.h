#ifndef KECC_C_IR_GENERATOR_H
#define KECC_C_IR_GENERATOR_H

#include "kecc/c/Clang.h"
#include "kecc/c/Diag.h"
#include "kecc/c/TypeConverter.h"
#include "kecc/c/Visitor.h"
#include "kecc/ir/Context.h"
#include "kecc/ir/IR.h"
#include "kecc/ir/IRAttributes.h"
#include "kecc/ir/IRBuilder.h"
#include "kecc/ir/Value.h"
#include "clang/AST/ASTContext.h"
#include "llvm/Support/SMLoc.h"
#include <cstddef>
#include <format>

namespace kecc::c {

class RecordDeclManager {
public:
  llvm::StringRef getRecordDeclID(const RecordDecl *decl) {
    auto it = recordDeclToIDMap.find(decl);
    if (it != recordDeclToIDMap.end()) {
      return it->second;
    }
    size_t id = nextID++;
    return recordDeclToIDMap[decl] = std::format("%t{}", id);
  }

  llvm::StringRef lookupRecordDeclID(const RecordDecl *decl) const {
    auto it = recordDeclToIDMap.find(decl);
    if (it != recordDeclToIDMap.end()) {
      return it->second;
    }
    llvm::report_fatal_error("RecordDecl not found");
  }

  bool hasRecordDeclID(const RecordDecl *decl) const {
    return recordDeclToIDMap.contains(decl);
  }

  void setRecordDeclID(const RecordDecl *decl) {
    assert(!decl->getName().empty() && "Only named struct/union is supported");
    recordDeclToIDMap[decl] = decl->getName().str();
  }

private:
  llvm::DenseMap<const RecordDecl *, std::string> recordDeclToIDMap;
  size_t nextID = 0;
};

class IRGenEnv {
public:
  struct Scope {
    Scope(IRGenEnv &env) : env(env) { env.varEnvStack.emplace_back(); }
    ~Scope() { env.varEnvStack.pop_back(); }

  private:
    IRGenEnv &env;
  };

  ir::Value lookup(llvm::StringRef name) const {
    for (const auto &env : llvm::reverse(varEnvStack)) {
      auto it = env.find(name);
      if (it != env.end()) {
        return it->second;
      }
    }
    return nullptr;
  }

  void insert(llvm::StringRef name, ir::Value value) {
    assert(!varEnvStack.empty() && "No scope");
    auto &currentEnv = varEnvStack.back();
    assert(!currentEnv.contains(name) && "Variable already declared");
    currentEnv[name] = value;
  }

  bool isGlobalScope() const { return varEnvStack.size() == 1; }

private:
  llvm::SmallVector<llvm::DenseMap<llvm::StringRef, ir::Value>, 8> varEnvStack;
};

class IRGenerator;

class GlobalInitGenerator {
public:
  GlobalInitGenerator(IRGenerator *irgen) : irgen(irgen) {}

  ir::InitializerAttr Visit(const Expr *expr);

  ir::InitializerAttr VisitBinaryOperator(const BinaryOperator *expr);
  ir::InitializerAttr VisitUnaryOperator(const UnaryOperator *expr);
  ir::InitializerAttr
  VisitUnaryExprOrTypeTraitExpr(const UnaryExprOrTypeTraitExpr *expr);
  ir::InitializerAttr VisitIntegerLiteral(const IntegerLiteral *expr);
  ir::InitializerAttr VisitFloatingLiteral(const FloatingLiteral *expr);
  ir::InitializerAttr VisitParenExpr(const ParenExpr *expr);

  ir::InitializerAttr VisitCastExpr(const CastExpr *expr);
  ir::InitializerAttr VisitImplicitCastExpr(const ImplicitCastExpr *expr);
  ir::InitializerAttr VisitCStyleCastExpr(const CStyleCastExpr *expr);

  ir::InitializerAttr VisitInitListExpr(const InitListExpr *expr);
  ir::InitializerAttr VisitMemberExpr(const MemberExpr *expr);
  ir::InitializerAttr VisitConditionalOperator(const ConditionalOperator *expr);
  ir::InitializerAttr VisitCallExpr(const CallExpr *expr);
  ir::InitializerAttr VisitDeclRefExpr(const DeclRefExpr *expr);

private:
  IRGenerator *irgen;
};

class IRGenerator : public StmtVisitor<IRGenerator>,
                    public DeclVisitor<IRGenerator> {
public:
  IRGenerator(clang::ASTContext *astCtx, ClangDiagManager *diags, ir::IR *ir,
              ir::IRContext *ctx, RecordDeclManager &recordDeclMgr)
      : StmtVisitor(diags), DeclVisitor(diags), astCtx(astCtx), diags(diags),
        ir(ir), ctx(ctx), recordDeclMgr(recordDeclMgr),
        typeConverter(diags, ctx, &recordDeclMgr), builder(ctx) {}

  bool generate(const TranslationUnitDecl *decl);

  void VisitCompoundStmt(const CompoundStmt *stmt);
  void VisitDeclStmt(const DeclStmt *stmt);
  void VisitReturnStmt(const ReturnStmt *stmt);
  void VisitIfStmt(const IfStmt *stmt);
  void VisitWhileStmt(const WhileStmt *stmt);
  void VisitForStmt(const ForStmt *stmt);
  void VisitSwitchStmt(const SwitchStmt *stmt);
  void VisitBreakStmt(const BreakStmt *stmt);
  void VisitContinueStmt(const ContinueStmt *stmt);
  void VisitNullStmt(const NullStmt *stmt);
  void VisitExpr(const Expr *expr);

  void VisitBinaryOperator(const BinaryOperator *expr);
  void VisitUnaryOperator(const UnaryOperator *expr);
  void VisitUnaryExprOrTypeTraitExpr(const UnaryExprOrTypeTraitExpr *expr);
  void VisitIntegerLiteral(const IntegerLiteral *expr);
  void VisitFloatingLiteral(const FloatingLiteral *expr);
  void VisitParenExpr(const ParenExpr *expr);

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
  void VisitValueDecl(const ValueDecl *decl);
  void VisitFunctionDecl(const FunctionDecl *decl);
  void VisitFieldDecl(const FieldDecl *decl);
  void VisitRecordDecl(const RecordDecl *decl);
  void VisitTypedefDecl(const TypedefDecl *decl);

  llvm::SMRange convertRange(const SourceRange &range) {
    return {llvm::SMLoc::getFromPointer(
                astCtx->getSourceManager().getCharacterData(range.getBegin())),
            llvm::SMLoc::getFromPointer(
                astCtx->getSourceManager().getCharacterData(range.getEnd()))};
  }

  llvm::SMRange getRange(const clang::IdentifierInfo *ident) {
    auto &sm = astCtx->getSourceManager();
    auto data = ident->getNameStart();
    auto end = data + ident->getLength();
    return {llvm::SMLoc::getFromPointer(data),
            llvm::SMLoc::getFromPointer(end)};
  }

private:
  friend class GlobalInitGenerator;

  clang::ASTContext *astCtx;
  ClangDiagManager *diags;
  ir::IRContext *ctx;
  RecordDeclManager &recordDeclMgr;
  TypeConverter typeConverter;
  IRGenEnv env;
  ir::IRBuilder builder;
  std::unique_ptr<ir::IR> ir;
};

} // namespace kecc::c

#endif // KECC_C_IR_GENERATOR_H
