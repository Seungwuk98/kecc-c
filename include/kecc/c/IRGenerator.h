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
#include "kecc/ir/Instruction.h"
#include "kecc/ir/Module.h"
#include "kecc/ir/TypeAttributeSupport.h"
#include "kecc/ir/Value.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/SMLoc.h"
#include <format>

namespace kecc::c {

class IRGenerator;

class FieldIndexMap {
public:
  FieldIndexMap() = default;
  FieldIndexMap(llvm::StringRef structName,
                const llvm::DenseMap<llvm::StringRef, size_t> &map,
                const llvm::DenseMap<size_t, const FieldIndexMap *> &anon)
      : structName(structName), fieldIndices(map), anonIndexMap(anon) {}

  std::vector<size_t> getFieldIndex(llvm::StringRef fieldName) const;

  const FieldIndexMap *getAnonFieldIndexMap(size_t index) const {
    auto it = anonIndexMap.find(index);
    if (it != anonIndexMap.end()) {
      return it->second;
    }
    return nullptr;
  }

  llvm::StringRef getStructName() const { return structName; }

private:
  llvm::StringRef structName;
  llvm::DenseMap<llvm::StringRef, size_t> fieldIndices;
  llvm::DenseMap<size_t, const FieldIndexMap *> anonIndexMap;
};

using StructFieldIndexMap = llvm::DenseMap<llvm::StringRef, FieldIndexMap>;

class RecordDeclManager {
public:
  RecordDeclManager() = default;

  llvm::StringRef getOrCreateRecordDeclID(const RecordDecl *decl,
                                          TypeConverter &typeConverter) {
    auto it = recordDeclToIDMap.find(decl);
    if (it != recordDeclToIDMap.end()) {
      return it->second;
    }
    if (!decl->getName().empty())
      return recordDeclToIDMap[decl] = decl->getName().str();

    std::string newName = std::format("%t{}", nextID++);
    return recordDeclToIDMap[decl] = newName;
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

  ir::StructSizeMap &getStructSizeMap() { return structSizeMap; }
  ir::StructFieldsMap &getStructFieldsMap() { return structFieldsMap; }
  StructFieldIndexMap &getStructFieldIndexMap() { return structFieldIndexMap; }
  const FieldIndexMap &getFieldIndexMap(llvm::StringRef structName) const {
    auto it = structFieldIndexMap.find(structName);
    if (it != structFieldIndexMap.end()) {
      return it->second;
    }
    llvm::report_fatal_error("Struct not found");
  }

  void updateStructSizeAndFields(IRGenerator *irgen, const RecordDecl *decl,
                                 TypeConverter &typeConverter);
  void updateStructSizeAndFields(
      IRGenerator *irgen, llvm::StringRef structName,
      llvm::ArrayRef<std::pair<llvm::StringRef, ir::Type>> fields,
      llvm::SMRange range = {});

private:
  llvm::DenseMap<const RecordDecl *, std::string> recordDeclToIDMap;
  ir::StructSizeMap structSizeMap;
  ir::StructFieldsMap structFieldsMap;
  StructFieldIndexMap structFieldIndexMap;

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
  struct Result {
    Result() = default;
    Result(llvm::APSInt intVal, SourceRange range)
        : value(intVal), range(range) {}
    Result(llvm::APInt intVal, SourceRange range)
        : value(llvm::APSInt(intVal)), range(range) {}
    Result(llvm::APInt intVal, bool isUnsigned, SourceRange range)
        : value(llvm::APSInt(intVal, isUnsigned)), range(range) {}
    Result(llvm::APFloat floatVal, SourceRange range)
        : value(floatVal), range(range) {}
    Result(std::vector<std::unique_ptr<Result>> listVal, SourceRange range)
        : value(std::move(listVal)), range(range) {}
    Result(std::nullopt_t) : value(std::monostate{}) {}
    Result(std::monostate) : value(std::monostate{}) {}

    Result(const Result &other) = delete;
    Result &operator=(const Result &other) = delete;
    Result(Result &&other) noexcept
        : value(std::move(other.value)), range(other.range) {}
    Result &operator=(Result &&other) noexcept {
      if (this != &other) {
        value = std::move(other.value);
        range = other.range;
      }
      return *this;
    }

    bool isValid() const {
      return !std::holds_alternative<std::monostate>(value);
    }

    operator bool() const { return isValid(); }

    bool isValueList() const {
      return std::holds_alternative<std::vector<std::unique_ptr<Result>>>(
          value);
    }
    bool isAPSInt() const {
      return std::holds_alternative<llvm::APSInt>(value);
    }
    bool isAPFloat() const {
      return std::holds_alternative<llvm::APFloat>(value);
    }
    llvm::APSInt getAPSInt() const { return std::get<llvm::APSInt>(value); }
    llvm::APFloat getAPFloat() const { return std::get<llvm::APFloat>(value); }
    const std::vector<std::unique_ptr<Result>> &getValueList() const {
      return std::get<std::vector<std::unique_ptr<Result>>>(value);
    }

    static Result fromAPSInt(llvm::APSInt intVal, SourceRange range) {
      return {intVal, range};
    }
    static Result fromInt(std::int64_t intVal, unsigned bitWidth, bool isSigned,
                          SourceRange range);
    static Result fromInt(std::int64_t intVal, ir::IntT intType,
                          SourceRange range);
    static Result fromAPFloat(llvm::APFloat floatVal, SourceRange range) {
      return {floatVal, range};
    }

    ir::InitializerAttr toInitializerAttr(IRGenerator *irgen);

    std::variant<std::monostate, std::vector<std::unique_ptr<Result>>,
                 llvm::APSInt, llvm::APFloat>
        value;
    SourceRange range;
  };

  GlobalInitGenerator(IRGenerator *irgen) : irgen(irgen) {}

  Result Visit(const Expr *expr);

  Result VisitBinaryOperator(const BinaryOperator *expr);
  Result VisitUnaryOperator(const UnaryOperator *expr);
  Result VisitUnaryExprOrTypeTraitExpr(const UnaryExprOrTypeTraitExpr *expr);
  Result VisitIntegerLiteral(const IntegerLiteral *expr);
  Result VisitFloatingLiteral(const FloatingLiteral *expr);
  Result VisitCharacterLiteral(const CharacterLiteral *expr);
  Result VisitParenExpr(const ParenExpr *expr);

  Result VisitCastExpr(const CastExpr *expr);

  Result VisitInitListExpr(const InitListExpr *expr);
  Result VisitConditionalOperator(const ConditionalOperator *expr);

  Result VisitBinMulOp(const BinaryOperator *expr);
  Result VisitBinAddOp(const BinaryOperator *expr);
  Result VisitBinSubOp(const BinaryOperator *expr);
  Result VisitBinDivOp(const BinaryOperator *expr);
  Result VisitBinRemOp(const BinaryOperator *expr);
  Result VisitBinShlOp(const BinaryOperator *expr);
  Result VisitBinShrOp(const BinaryOperator *expr);
  Result VisitBinLTOp(const BinaryOperator *expr);
  Result VisitBinGTOp(const BinaryOperator *expr);
  Result VisitBinLEOp(const BinaryOperator *expr);
  Result VisitBinGEOp(const BinaryOperator *expr);
  Result VisitBinEQOp(const BinaryOperator *expr);
  Result VisitBinNEOp(const BinaryOperator *expr);
  Result VisitBinAndOp(const BinaryOperator *expr);
  Result VisitBinXorOp(const BinaryOperator *expr);
  Result VisitBinOrOp(const BinaryOperator *expr);
  Result VisitBinLAndOp(const BinaryOperator *expr);
  Result VisitBinLOrOp(const BinaryOperator *expr);
  Result VisitBinCommaOp(const BinaryOperator *expr);

  Result VisitUnPlusOp(const UnaryOperator *expr);
  Result VisitUnMinusOp(const UnaryOperator *expr);
  Result VisitUnNotOp(const UnaryOperator *expr);
  Result VisitUnLNotOp(const UnaryOperator *expr);

private:
  IRGenerator *irgen;
};

class ExprEvaluator;

class LocalInitGenerator {
public:
  LocalInitGenerator(IRGenerator *irgen, ir::IRBuilder &builder,
                     ir::Value memory);

  struct MemoryGuard {
    MemoryGuard(LocalInitGenerator &gen, ir::Value memory)
        : gen(gen), savedMemory(gen.memory) {
      gen.memory = memory;
      gen.memoryInnerT = memory.getType().cast<ir::PointerT>().getPointeeType();
    }
    ~MemoryGuard() {
      gen.memory = savedMemory;
      gen.memoryInnerT =
          savedMemory.getType().cast<ir::PointerT>().getPointeeType();
    }

  private:
    LocalInitGenerator &gen;
    ir::Value savedMemory;
  };

  void Visit(const Expr *expr);
  void VisitInitListExpr(const InitListExpr *expr);

private:
  IRGenerator *irgen;
  ir::IRBuilder &builder;
  ir::Value memory;
  ir::Type memoryInnerT;
};

class ExprEvaluator {
public:
  ExprEvaluator(IRGenerator *irgen);
  ir::Value Visit(const Expr *expr);

  ir::Value VisitBinaryOperator(const BinaryOperator *expr);
  ir::Value VisitUnaryOperator(const UnaryOperator *expr);
  ir::Value VisitUnaryExprOrTypeTraitExpr(const UnaryExprOrTypeTraitExpr *expr);
  ir::Value VisitIntegerLiteral(const IntegerLiteral *expr);
  ir::Value VisitFloatingLiteral(const FloatingLiteral *expr);
  ir::Value VisitCharacterLiteral(const CharacterLiteral *expr);
  ir::Value VisitParenExpr(const ParenExpr *expr);
  ir::Value VisitArraySubscriptExpr(const ArraySubscriptExpr *expr);

  ir::Value VisitCastExpr(const CastExpr *expr);

  ir::Value VisitMemberExpr(const MemberExpr *expr);
  ir::Value VisitConditionalOperator(const ConditionalOperator *expr);
  ir::Value VisitCallExpr(const CallExpr *expr);
  ir::Value VisitDeclRefExpr(const DeclRefExpr *expr);

  ir::Value VisitBinMulOp(const BinaryOperator *expr);
  ir::Value VisitBinAddOp(const BinaryOperator *expr);
  ir::Value VisitBinSubOp(const BinaryOperator *expr);
  ir::Value VisitBinDivOp(const BinaryOperator *expr);
  ir::Value VisitBinRemOp(const BinaryOperator *expr);
  ir::Value VisitBinShlOp(const BinaryOperator *expr);
  ir::Value VisitBinShrOp(const BinaryOperator *expr);
  ir::Value VisitBinLTOp(const BinaryOperator *expr);
  ir::Value VisitBinGTOp(const BinaryOperator *expr);
  ir::Value VisitBinLEOp(const BinaryOperator *expr);
  ir::Value VisitBinGEOp(const BinaryOperator *expr);
  ir::Value VisitBinEQOp(const BinaryOperator *expr);
  ir::Value VisitBinNEOp(const BinaryOperator *expr);
  ir::Value VisitBinAndOp(const BinaryOperator *expr);
  ir::Value VisitBinXorOp(const BinaryOperator *expr);
  ir::Value VisitBinOrOp(const BinaryOperator *expr);
  ir::Value VisitBinLAndOp(const BinaryOperator *expr);
  ir::Value VisitBinLOrOp(const BinaryOperator *expr);
  ir::Value VisitBinCommaOp(const BinaryOperator *expr);
  ir::Value VisitBinAssignOp(const BinaryOperator *expr);
  ir::Value VisitBinMulAssignOp(const BinaryOperator *expr);
  ir::Value VisitBinDivAssignOp(const BinaryOperator *expr);
  ir::Value VisitBinRemAssignOp(const BinaryOperator *expr);
  ir::Value VisitBinAddAssignOp(const BinaryOperator *expr);
  ir::Value VisitBinSubAssignOp(const BinaryOperator *expr);
  ir::Value VisitBinShlAssignOp(const BinaryOperator *expr);
  ir::Value VisitBinShrAssignOp(const BinaryOperator *expr);
  ir::Value VisitBinAndAssignOp(const BinaryOperator *expr);
  ir::Value VisitBinXorAssignOp(const BinaryOperator *expr);
  ir::Value VisitBinOrAssignOp(const BinaryOperator *expr);

  ir::Value VisitUnPlusOp(const UnaryOperator *expr);
  ir::Value VisitUnMinusOp(const UnaryOperator *expr);
  ir::Value VisitUnNotOp(const UnaryOperator *expr);
  ir::Value VisitUnLNotOp(const UnaryOperator *expr);
  ir::Value VisitUnPreIncOp(const UnaryOperator *expr);
  ir::Value VisitUnPreDecOp(const UnaryOperator *expr);
  ir::Value VisitUnPostIncOp(const UnaryOperator *expr);
  ir::Value VisitUnPostDecOp(const UnaryOperator *expr);
  ir::Value VisitUnAddrOfOp(const UnaryOperator *expr);
  ir::Value VisitUnDerefOp(const UnaryOperator *expr);

private:
  ir::Value getIntegerValue(std::int64_t val, ir::IntT intT);

  IRGenerator *irgen;
  ir::IRBuilder &builder;
};

class IRGenerator : public StmtVisitor<IRGenerator>,
                    public DeclVisitor<IRGenerator> {
public:
  IRGenerator(clang::ASTContext *astCtx, ClangDiagManager *diags,
              RecordDeclManager &recordDeclMgr, ir::IR *ir)
      : StmtVisitor(diags), DeclVisitor(diags), astCtx(astCtx), diags(diags),
        ctx(ir->getContext()), recordDeclMgr(recordDeclMgr),
        typeConverter(diags, ctx, &recordDeclMgr), builder(ctx),
        exprEvaluator(this), ir(ir) {}

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
  void VisitDoStmt(const DoStmt *stmt);
  void VisitExpr(const Expr *expr);

  ir::Value EvaluateExpr(const Expr *expr);

  void VisitTranslationUnitDecl(const TranslationUnitDecl *decl);
  void VisitVarDecl(const VarDecl *decl);
  void VisitFunctionDecl(const FunctionDecl *decl);
  void VisitRecordDecl(const RecordDecl *decl);
  void VisitTypedefDecl(const TypedefDecl *decl);

  llvm::SMRange convertRange(const SourceRange &range) {
    return {llvm::SMLoc::getFromPointer(
                astCtx->getSourceManager().getCharacterData(range.getBegin())),
            llvm::SMLoc::getFromPointer(
                astCtx->getSourceManager().getCharacterData(range.getEnd()))};
  }
  llvm::SMRange convertRange(const SourceLocation &loc) {
    return {llvm::SMLoc::getFromPointer(
                astCtx->getSourceManager().getCharacterData(loc)),
            llvm::SMLoc::getFromPointer(
                astCtx->getSourceManager().getCharacterData(loc) + 1)};
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
  friend class LocalInitGenerator;
  friend class ExprEvaluator;
  friend class RecordDeclManager;

  struct FunctionData {
    FunctionData(ir::Function *function, const FunctionDecl *decl)
        : function(function), decl(decl), blockCount(0) {}

    ir::Function *function;
    const FunctionDecl *decl;
    size_t blockCount;
  };

  ir::Block *createNewBlock() {
    assert(currentFunctionData && "Not in a function");
    auto block = currentFunctionData->function->addBlock(
        currentFunctionData->blockCount++);
    return block;
  }

  struct BreakPoint {
    BreakPoint(IRGenerator &irgen, ir::Block *breakBlock,
               ir::Block *continueBlock)
        : irgen(irgen), savedBreakBlock(irgen.breakJArg),
          savedContinueBlock(irgen.continueJArg) {
      irgen.breakJArg = ir::JumpArgState(breakBlock);
      irgen.continueJArg = ir::JumpArgState(continueBlock);
    }
    BreakPoint(IRGenerator &irgen, std::optional<ir::JumpArgState> breakJArg,
               std::optional<ir::JumpArgState> continueJArg)
        : irgen(irgen), savedBreakBlock(irgen.breakJArg),
          savedContinueBlock(irgen.continueJArg) {
      irgen.breakJArg = breakJArg;
      irgen.continueJArg = continueJArg;
    }
    ~BreakPoint() {
      irgen.breakJArg = savedBreakBlock;
      irgen.continueJArg = savedContinueBlock;
    }

  private:
    IRGenerator &irgen;
    std::optional<ir::JumpArgState> savedBreakBlock;
    std::optional<ir::JumpArgState> savedContinueBlock;
  };

  std::pair<llvm::StringRef, ir::Type> fieldInfo(const FieldDecl *field);

  clang::ASTContext *astCtx;
  ClangDiagManager *diags;
  ir::IRContext *ctx;
  RecordDeclManager &recordDeclMgr;
  TypeConverter typeConverter;
  IRGenEnv env;
  ir::IRBuilder builder;
  ExprEvaluator exprEvaluator;

  ir::IR *ir;
  std::unique_ptr<FunctionData> currentFunctionData = nullptr;

  std::optional<ir::JumpArgState> breakJArg = std::nullopt;
  std::optional<ir::JumpArgState> continueJArg = std::nullopt;
};

} // namespace kecc::c

#endif // KECC_C_IR_GENERATOR_H
