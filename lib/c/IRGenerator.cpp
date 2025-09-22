#include "kecc/c/IRGenerator.h"
#include "kecc/c/TypeConverter.h"
#include "kecc/c/Visitor.h"
#include "kecc/ir/IRAttributes.h"
#include "kecc/ir/IRBuilder.h"
#include "kecc/ir/IRInstructions.h"
#include "kecc/ir/IRTypes.h"
#include "kecc/ir/Instruction.h"
#include "clang/AST/Decl.h"
#include "clang/AST/OperationKinds.h"
#include "clang/AST/Stmt.h"

namespace kecc::c {

void IRGenerator::VisitTranslationUnitDecl(const TranslationUnitDecl *D) {
  IRGenEnv::Scope scope(env); // global scope
  ir = std::unique_ptr<ir::IR>(new ir::IR(ctx));

  for (auto I = D->decls_begin(), E = D->decls_end(); I != E; ++I) {
    if (const auto *typedefDecl = llvm::dyn_cast<TypedefDecl>(*I)) {
      if (typedefDecl->isImplicit())
        continue;
    }
    DeclVisitor::Visit(*I);
  }
}

void IRGenerator::VisitVarDecl(const VarDecl *D) {
  auto type =
      typeConverter.VisitQualType(D->getType(), D->getTypeSpecStartLoc());
  assert(type && "Variable type conversion failed");

  ir::IRBuilder::InsertionGuard guard(builder);

  if (env.isGlobalScope()) {
    builder.setInsertionPoint(ir->getGlobalBlock());
    ir::InitializerAttr init;
    if (D->hasInit()) {
      GlobalInitGenerator initGen(this);
      auto result = initGen.Visit(D->getInit());
      assert(result && "Global variable initializer generation failed");
      init = result.toInitializerAttr(this);
      assert(init && "Global variable initializer conversion failed");
    }
    builder.create<ir::inst::GlobalVariableDefinition>(
        convertRange(D->getSourceRange()), type, D->getName(), init);

    auto name = D->getName();
    assert(!name.empty() && "Unnamed global variable");

    builder.setInsertionPoint(ir->getConstantBlock());
    auto gv = builder.create<ir::inst::Constant>(
        getRange(D->getIdentifier()),
        ir::ConstantVariableAttr::get(ctx, name, ir::PointerT::get(ctx, type)));
    env.insert(name, gv);
  } else {
    auto name = D->getName();
    assert(!name.empty() && "Unnamed local variable");

    auto *currBlock = builder.getCurrentBlock();
    builder.setInsertionPoint(
        currBlock->getParentFunction()->getAllocationBlock());
    auto localVar = builder.create<ir::inst::LocalVariable>(
        convertRange(D->getSourceRange()), ir::PointerT::get(ctx, type));
    env.insert(name, localVar);
    if (D->hasInit()) {
      LocalInitGenerator initGen(this, builder, localVar);
      initGen.Visit(D->getInit());
    }
  }
}

void IRGenerator::VisitFunctionDecl(const FunctionDecl *D) {
  auto declType =
      typeConverter.VisitQualType(D->getType(), D->getTypeSpecStartLoc());
  assert(declType && "Function type conversion failed");
  assert(declType.isa<ir::FunctionT>() && "Function type expected");

  auto funcT = declType.cast<ir::FunctionT>();
  auto name = D->getName();

  ir::Function *func;
  if (!(func = ir->getFunction(D->getName()))) {
    func = new ir::Function(convertRange(D->getSourceRange()), name, funcT,
                            ir.get(), ctx);
    ir->addFunction(func);
  }

  auto funcPointerT = ir::PointerT::get(ctx, funcT);
  assert(env.isGlobalScope() && "Function declaration in non-global scope");
  {
    ir::IRBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(ir->getConstantBlock());
    auto funcConst = builder.create<ir::inst::Constant>(
        getRange(D->getIdentifier()),
        ir::ConstantVariableAttr::get(ctx, name, funcPointerT));
    env.insert(name, funcConst);
  }

  if (!D->hasBody())
    return;

  currentFunctionData = std::make_unique<FunctionData>(func, D);

  ir::Block *entryBlock = createNewBlock();
  func->setEntryBlock(entryBlock->getId());
  ir::IRBuilder::InsertionGuard guard(builder);

  assert(!func->hasDefinition() && "Function redefinition");

  llvm::ArrayRef<ParmVarDecl *> funcParams = D->parameters();
  IRGenEnv::Scope scope(env); // function scope
  llvm::SmallVector<std::pair<ir::inst::LocalVariable, ir::Phi>> paramValues;
  for (size_t i = 0; i < funcParams.size(); ++i) {
    ParmVarDecl *param = funcParams[i];
    ir::Type paramType = funcT.getArgTypes()[i];
    auto paramName = param->getName();

    builder.setInsertionPoint(func->getAllocationBlock());
    auto localVar = builder.create<ir::inst::LocalVariable>(
        convertRange(param->getSourceRange()),
        ir::PointerT::get(ctx, paramType));
    localVar.setValueName(paramName);
    builder.setInsertionPoint(entryBlock);
    auto phi = builder.create<ir::Phi>(convertRange(param->getSourceRange()),
                                       paramType.constCanonicalize());

    if (!paramName.empty()) {
      env.insert(paramName, localVar);
      localVar.setValueName(paramName);
      phi.setValueName(paramName);
    }
  }

  for (const auto &[localVar, phi] : paramValues) {
    builder.setInsertionPoint(entryBlock);
    builder.create<ir::inst::Store>(localVar->getRange(), phi, localVar);
  }

  StmtVisitor::Visit(D->getBody());

  currentFunctionData.reset();
}

void IRGenerator::VisitRecordDecl(const RecordDecl *D) {
  if (!D->hasBody())
    return;

  llvm::StringRef structName;
  if (D->getName().empty())
    structName = recordDeclMgr.getRecordDeclID(D);
  else
    structName = D->getName();

  llvm::SmallVector<std::pair<llvm::StringRef, ir::Type>> fields;

  for (const auto *field : D->fields()) {
    auto [fieldName, fieldType] = fieldInfo(field);
    fields.emplace_back(fieldName, fieldType);
  }

  ir::IRBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(ir->getStructBlock());

  recordDeclMgr.updateStructSizeAndFields(structName, fields);
  builder.create<ir::inst::StructDefinition>(convertRange(D->getSourceRange()),
                                             fields, structName);
}

void IRGenerator::VisitTypedefDecl(const TypedefDecl *D) {
  // Just visit the underlying type to ensure it's processed.
  typeConverter.VisitQualType(D->getUnderlyingType(), D->getLocation());
}

void IRGenerator::VisitCompoundStmt(const CompoundStmt *S) {
  IRGenEnv::Scope scope(env);
  for (const auto *stmt : S->body()) {
    StmtVisitor::Visit(stmt);
  }
}

void IRGenerator::VisitDeclStmt(const clang::DeclStmt *S) {
  for (const auto *decl : S->decls()) {
    DeclVisitor::Visit(decl);
  }
}

void IRGenerator::VisitReturnStmt(const ReturnStmt *S) {
  ir::Value value;
  if (const Expr *returnExpr = S->getRetValue()) {
    value = EvaluateExpr(returnExpr);
    assert(value && "Return expression generation failed");
  } else {
    ir::IRBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(ir->getConstantBlock());
    value = builder.create<ir::inst::Constant>(
        convertRange(S->getSourceRange()), ir::ConstantUnitAttr::get(ctx));
  }
  builder.create<ir::inst::Return>(convertRange(S->getSourceRange()), value);
}

void IRGenerator::VisitIfStmt(const IfStmt *S) {
  ir::Value condV = EvaluateExpr(S->getCond());
  assert(condV && "If condition generation failed");

  ir::Type condT = condV.getType();

  if (!condT.isBoolean()) {
    auto boolType = ir::IntT::get(ctx, 1, false);

    condV = builder.create<ir::inst::TypeCast>(
        convertRange(S->getCond()->getSourceRange()), condV, boolType);
  }

  ir::Block *thenBlock = createNewBlock();
  ir::Block *elseBlock = createNewBlock();
  ir::Block *mergeBlock = createNewBlock();

  ir::JumpArgState thenJArg(thenBlock), elseJArg(elseBlock),
      mergeJArg(mergeBlock);

  builder.create<ir::inst::Branch>(convertRange(S->getBeginLoc()), condV,
                                   thenJArg, elseJArg);

  builder.setInsertionPoint(thenBlock);
  StmtVisitor::Visit(S->getThen());
  builder.create<ir::inst::Jump>(convertRange(S->getThen()->getEndLoc()),
                                 mergeJArg);

  if (S->getElse()) {
    builder.setInsertionPoint(elseBlock);
    StmtVisitor::Visit(S->getElse());
    builder.create<ir::inst::Jump>(convertRange(S->getElse()->getEndLoc()),
                                   mergeJArg);
  } else
    builder.create<ir::inst::Jump>(convertRange(S->getEndLoc()), mergeJArg);

  builder.setInsertionPoint(mergeBlock);
}

void IRGenerator::VisitWhileStmt(const WhileStmt *S) {
  ir::Block *condBlock = createNewBlock();
  ir::Block *bodyBlock = createNewBlock();
  ir::Block *mergeBlock = createNewBlock();

  ir::JumpArgState condJArg(condBlock), bodyJArg(bodyBlock),
      mergeJArg(mergeBlock);

  builder.create<ir::inst::Jump>(convertRange(S->getCond()->getBeginLoc()),
                                 condJArg);

  builder.setInsertionPoint(condBlock);
  ir::Value condV = EvaluateExpr(S->getCond());
  assert(condV && "While condition generation failed");

  ir::Type condT = condV.getType();
  if (!condT.isBoolean()) {
    auto boolType = ir::IntT::get(ctx, 1, false);

    condV = builder.create<ir::inst::TypeCast>(
        convertRange(S->getCond()->getEndLoc()), condV, boolType);
  }

  builder.create<ir::inst::Branch>(convertRange(S->getCond()->getEndLoc()),
                                   condV, bodyJArg, mergeJArg);

  builder.setInsertionPoint(bodyBlock);
  {
    BreakPoint bp(*this, mergeBlock, condBlock);
    StmtVisitor::Visit(S->getBody());
  }

  builder.create<ir::inst::Jump>(convertRange(S->getBody()->getEndLoc()),
                                 condJArg);

  builder.setInsertionPoint(mergeBlock);
}

void IRGenerator::VisitForStmt(const ForStmt *S) {
  IRGenEnv::Scope scope(env);
  ir::Block *condBlock = createNewBlock();
  ir::Block *bodyBlock = createNewBlock();
  ir::Block *incBlock = createNewBlock();
  ir::Block *mergeBlock = createNewBlock();

  ir::JumpArgState condJArg(condBlock), bodyJArg(bodyBlock), incJArg(incBlock),
      mergeJArg(mergeBlock);

  if (S->getInit()) {
    StmtVisitor::Visit(S->getInit());
    builder.create<ir::inst::Jump>(convertRange(S->getInit()->getBeginLoc()),
                                   condJArg);
  } else
    builder.create<ir::inst::Jump>(convertRange(S->getBeginLoc()), condJArg);

  builder.setInsertionPoint(condBlock);

  if (auto *cond = S->getCond()) {
    ir::Value condV = EvaluateExpr(cond);
    assert(condV && "For condition generation failed");

    ir::Type condT = condV.getType();
    if (!condT.isBoolean()) {
      auto boolType = ir::IntT::get(ctx, 1, false);

      condV = builder.create<ir::inst::TypeCast>(
          convertRange(cond->getEndLoc()), condV, boolType);
    }

    builder.create<ir::inst::Branch>(convertRange(cond->getEndLoc()), condV,
                                     bodyJArg, mergeJArg);
  } else
    builder.create<ir::inst::Jump>(convertRange(S->getBeginLoc()), bodyJArg);

  builder.setInsertionPoint(bodyBlock);
  {
    BreakPoint bp(*this, mergeBlock, incBlock);
    StmtVisitor::Visit(S->getBody());
  }

  builder.create<ir::inst::Jump>(
      convertRange(S->getBody()->getSourceRange().getEnd()), incJArg);

  builder.setInsertionPoint(incBlock);
  if (S->getInc()) {
    StmtVisitor::Visit(S->getInc());
  }
  builder.create<ir::inst::Jump>(convertRange(S->getRParenLoc()), condJArg);

  builder.setInsertionPoint(mergeBlock);
}

void IRGenerator::VisitDoStmt(const DoStmt *S) {
  ir::Block *bodyBlock = createNewBlock();
  ir::Block *condBlock = createNewBlock();
  ir::Block *mergeBlock = createNewBlock();

  ir::JumpArgState bodyJArg(bodyBlock), condJArg(condBlock),
      mergeJArg(mergeBlock);

  builder.create<ir::inst::Jump>(convertRange(S->getBeginLoc()), bodyJArg);

  builder.setInsertionPoint(bodyBlock);
  {
    BreakPoint bp(*this, mergeBlock, condBlock);
    StmtVisitor::Visit(S->getBody());
  }
  builder.create<ir::inst::Jump>(
      convertRange(S->getBody()->getSourceRange().getEnd()), condJArg);

  builder.setInsertionPoint(condBlock);
  ir::Value condV = EvaluateExpr(S->getCond());
  assert(condV && "Do-while condition generation failed");

  ir::Type condT = condV.getType();
  if (!condT.isBoolean()) {
    auto boolType = ir::IntT::get(ctx, 1, false);

    condV = builder.create<ir::inst::TypeCast>(
        convertRange(S->getCond()->getSourceRange()), condV, boolType);
  }

  builder.create<ir::inst::Branch>(convertRange(S->getRParenLoc()), condV,
                                   bodyJArg, mergeJArg);

  builder.setInsertionPoint(mergeBlock);
}

void IRGenerator::VisitSwitchStmt(const SwitchStmt *S) {
  ir::Value condV = EvaluateExpr(S->getCond());
  assert(condV && "Switch condition generation failed");

  ir::Block *mergeBlock = createNewBlock();

  ir::Block *defaultBlock = createNewBlock();
  ir::JumpArgState defaultJArg(defaultBlock), mergeJArg(mergeBlock);

  BreakPoint bp(*this, mergeBlock, nullptr);

  llvm::SmallVector<ir::Value> caseValues;
  llvm::SmallVector<ir::JumpArgState> jumpArgs;
  llvm::SmallVector<ir::Block *> caseBlocks;

  const CompoundStmt *body = llvm::cast<CompoundStmt>(S->getBody());

  bool first = true;
  // we don't add scope here because switch doesn't introduce a new scope
  for (const Stmt *stmt : body->body()) {
    if (const CaseStmt *caseStmt = llvm::dyn_cast<CaseStmt>(stmt)) {
      ir::Value caseV = EvaluateExpr(caseStmt->getLHS());
      assert(caseV && "Case value generation failed");

      ir::Block *caseBlock = createNewBlock();
      caseValues.emplace_back(caseV);
      caseBlocks.emplace_back(caseBlock);
      jumpArgs.emplace_back(caseBlock);
      if (!first) {
        builder.create<ir::inst::Jump>(convertRange(caseStmt->getEndLoc()),
                                       caseBlocks.back());
      } else
        first = false;
      builder.setInsertionPoint(caseBlock);

    } else if (const DefaultStmt *defaultStmt =
                   llvm::dyn_cast<DefaultStmt>(stmt)) {
      if (!first) {
        builder.create<ir::inst::Jump>(convertRange(defaultStmt->getEndLoc()),
                                       defaultBlock);
      } else
        first = false;
      builder.setInsertionPoint(defaultBlock);
    } else {
      StmtVisitor::Visit(stmt);
    }
  }
}

void IRGenerator::VisitBreakStmt(const BreakStmt *S) {
  assert(breakJArg && "Break statement not within a loop or switch");
  builder.create<ir::inst::Jump>(convertRange(S->getSourceRange()), *breakJArg);
  builder.setInsertionPoint(breakJArg->getBlock());
}

void IRGenerator::VisitContinueStmt(const ContinueStmt *S) {
  assert(continueJArg && "Continue statement not within a loop");
  builder.create<ir::inst::Jump>(convertRange(S->getSourceRange()),
                                 *continueJArg);
  builder.setInsertionPoint(continueJArg->getBlock());
}

void IRGenerator::VisitNullStmt(const NullStmt *S) {
  // Do nothing for null statements
}

void IRGenerator::VisitExpr(const Expr *S) { (void)EvaluateExpr(S); }

ir::Value IRGenerator::EvaluateExpr(const Expr *E) {
  return exprEvaluator.Visit(E);
}

LocalInitGenerator::LocalInitGenerator(IRGenerator *irgen,
                                       ir::IRBuilder &builder, ir::Value memory)
    : irgen(irgen), builder(builder), memory(memory),
      memoryInnerT(memory.getType().cast<ir::PointerT>().getPointeeType()) {}

void LocalInitGenerator::Visit(const Expr *expr) {
  if (const InitListExpr *initExpr = llvm::dyn_cast<InitListExpr>(expr)) {
    VisitInitListExpr(initExpr);
  } else {
    ir::Value value = irgen->EvaluateExpr(expr);
    assert(value && "Expression evaluation failed");
    if (value.getType() != memoryInnerT) {
      value = builder.create<ir::inst::TypeCast>(
          irgen->convertRange(expr->getEndLoc()), value, memoryInnerT);
    }

    builder.create<ir::inst::Store>(irgen->convertRange(expr->getEndLoc()),
                                    value, memory);
  }
}

void LocalInitGenerator::VisitInitListExpr(const InitListExpr *expr) {
  if (ir::ArrayT arrayT = memoryInnerT.dyn_cast<ir::ArrayT>()) {
    size_t numInits = expr->getNumInits();
    size_t arraySize = arrayT.getSize();
    ir::Type elemType = arrayT.getElementType();
    ir::Type elemPtrType = ir::PointerT::get(irgen->ctx, elemType);

    assert(numInits <= arraySize && "Too many initializers for array type");

    const auto &[size, _] =
        elemType.getSizeAndAlign(irgen->recordDeclMgr.getStructSizeMap());

    for (size_t i = 0; i < numInits; ++i) {
      ir::Value newMemory;
      {
        ir::IRBuilder::InsertionGuard guard(builder);
        builder.setInsertionPoint(irgen->ir->getConstantBlock());
        ir::Value index = builder.create<ir::inst::Constant>(
            irgen->convertRange(expr->getInit(i)->getSourceRange()),
            ir::ConstantIntAttr::get(irgen->ctx, i * size,
                                     irgen->ctx->getArchitectureBitSize(),
                                     true));
        newMemory = builder.create<ir::inst::Gep>(
            irgen->convertRange(expr->getInit(i)->getSourceRange()), memory,
            index, elemPtrType);
      }
      MemoryGuard mg(*this, newMemory);
      Visit(expr->getInit(i));
    }
  } else if (ir::NameStruct structT = memoryInnerT.dyn_cast<ir::NameStruct>()) {
    const auto &[size, align, offsets] =
        irgen->recordDeclMgr.getStructSizeMap().at(structT.getName());
    size_t numInits = expr->getNumInits();
    size_t fieldSize = offsets.size();

    assert(numInits <= fieldSize && "Too many initializers for struct type");

    for (size_t i = 0; i < numInits; ++i) {
      ir::Value newMemory;
      {
        ir::IRBuilder::InsertionGuard guard(builder);
        builder.setInsertionPoint(irgen->ir->getConstantBlock());
        ir::Value index = builder.create<ir::inst::Constant>(
            irgen->convertRange(expr->getInit(i)->getSourceRange()),
            ir::ConstantIntAttr::get(irgen->ctx, offsets[i],
                                     irgen->ctx->getArchitectureBitSize(),
                                     true));
        newMemory = builder.create<ir::inst::Gep>(
            irgen->convertRange(expr->getInit(i)->getSourceRange()), memory,
            index, ir::PointerT::get(irgen->ctx, structT));
      }
      MemoryGuard mg(*this, newMemory);
      Visit(expr->getInit(i));
    }
  } else if (expr->getNumInits() == 1) {
    Visit(expr->getInit(0));
  } else if (expr->getNumInits() == 0) {
    // Do nothing for empty initializer list
  } else {
    llvm_unreachable("Initializer list for non-aggregate type");
  }
}

GlobalInitGenerator::Result GlobalInitGenerator::Visit(const Expr *expr) {
  switch (expr->getStmtClass()) {
  case Stmt::BinaryOperatorClass:
    return VisitBinaryOperator(llvm::cast<BinaryOperator>(expr));
  case Stmt::UnaryOperatorClass:
    return VisitUnaryOperator(llvm::cast<UnaryOperator>(expr));
  case Stmt::UnaryExprOrTypeTraitExprClass:
    return VisitUnaryExprOrTypeTraitExpr(
        llvm::cast<UnaryExprOrTypeTraitExpr>(expr));
  case Stmt::IntegerLiteralClass:
    return VisitIntegerLiteral(llvm::cast<IntegerLiteral>(expr));
  case Stmt::FloatingLiteralClass:
    return VisitFloatingLiteral(llvm::cast<FloatingLiteral>(expr));
  case Stmt::CharacterLiteralClass:
    return VisitCharacterLiteral(llvm::cast<CharacterLiteral>(expr));
  case Stmt::ParenExprClass:
    return VisitParenExpr(llvm::cast<ParenExpr>(expr));
  case Stmt::ImplicitCastExprClass:
    return VisitCastExpr(llvm::cast<ImplicitCastExpr>(expr));
  case Stmt::CStyleCastExprClass:
    return VisitCastExpr(llvm::cast<CStyleCastExpr>(expr));
  case Stmt::InitListExprClass:
    return VisitInitListExpr(llvm::cast<InitListExpr>(expr));
  case Stmt::ConditionalOperatorClass:
    return VisitConditionalOperator(llvm::cast<ConditionalOperator>(expr));
  default:
    llvm_unreachable("Unsupported expression in global initializer");
  }
}

GlobalInitGenerator::Result
GlobalInitGenerator::VisitBinaryOperator(const BinaryOperator *expr) {
  switch (expr->getOpcode()) {
  case clang::BO_Mul:
    return VisitBinMulOp(expr);
  case clang::BO_Div:
    return VisitBinDivOp(expr);
  case clang::BO_Rem:
    return VisitBinRemOp(expr);
  case clang::BO_Add:
    return VisitBinAddOp(expr);
  case clang::BO_Sub:
    return VisitBinSubOp(expr);
  case clang::BO_Shl:
    return VisitBinShlOp(expr);
  case clang::BO_Shr:
    return VisitBinShrOp(expr);
  case clang::BO_LT:
    return VisitBinLTOp(expr);
  case clang::BO_GT:
    return VisitBinGTOp(expr);
  case clang::BO_LE:
    return VisitBinLEOp(expr);
  case clang::BO_GE:
    return VisitBinGEOp(expr);
  case clang::BO_EQ:
    return VisitBinEQOp(expr);
  case clang::BO_NE:
    return VisitBinNEOp(expr);
  case clang::BO_And:
    return VisitBinAndOp(expr);
  case clang::BO_Xor:
    return VisitBinXorOp(expr);
  case clang::BO_Or:
    return VisitBinOrOp(expr);
  case clang::BO_LAnd:
    return VisitBinLAndOp(expr);
  case clang::BO_LOr:
    return VisitBinLOrOp(expr);
  case clang::BO_Comma:
    return VisitBinCommaOp(expr);
  default:
    llvm_unreachable("Unsupported binary operator in global initializer");
  }
}

GlobalInitGenerator::Result
GlobalInitGenerator::VisitUnaryOperator(const UnaryOperator *expr) {
  switch (expr->getOpcode()) {
  case clang::UO_Plus:
    return VisitUnPlusOp(expr);
  case clang::UO_Minus:
    return VisitUnMinusOp(expr);
  case clang::UO_Not:
    return VisitUnNotOp(expr);
  case clang::UO_LNot:
    return VisitUnLNotOp(expr);
  default:
    llvm_unreachable("Unsupported unary operator in global initializer");
  }
}

GlobalInitGenerator::Result GlobalInitGenerator::VisitUnaryExprOrTypeTraitExpr(
    const UnaryExprOrTypeTraitExpr *expr) {
  size_t value;
  switch (expr->getKind()) {
  case clang::UETT_SizeOf: {
    ir::Type type;

    if (expr->isArgumentType()) {
      type = irgen->typeConverter.VisitQualType(
          expr->getArgumentType(),
          expr->getArgumentTypeInfo()->getTypeLoc().getBeginLoc());
    } else {
      type = irgen->typeConverter.VisitQualType(
          expr->getArgumentExpr()->getType(), expr->getBeginLoc());
    }
    value = type.getSizeAndAlign(irgen->recordDeclMgr.getStructSizeMap()).first;
  }
  case clang::UETT_AlignOf: {
    ir::Type type;
    if (expr->isArgumentType()) {
      type = irgen->typeConverter.VisitQualType(
          expr->getArgumentType(),
          expr->getArgumentTypeInfo()->getTypeLoc().getBeginLoc());
    } else {
      type = irgen->typeConverter.VisitQualType(
          expr->getArgumentExpr()->getType(), expr->getBeginLoc());
    }
    value =
        type.getSizeAndAlign(irgen->recordDeclMgr.getStructSizeMap()).second;
  }
  default:
    llvm_unreachable("Unsupported unary expr or type trait in global init");
  }
  return Result::fromInt(
      value,
      irgen->typeConverter.VisitQualType(expr->getType(), expr->getBeginLoc())
          .cast<ir::IntT>(),
      expr->getSourceRange());
}

GlobalInitGenerator::Result
GlobalInitGenerator::VisitIntegerLiteral(const clang::IntegerLiteral *expr) {
  llvm::APInt value = expr->getValue();
  return {value, expr->getType()->isUnsignedIntegerType(),
          expr->getSourceRange()};
}

GlobalInitGenerator::Result
GlobalInitGenerator::VisitFloatingLiteral(const FloatingLiteral *expr) {
  llvm::APFloat value = expr->getValue();
  return {value, expr->getSourceRange()};
}

GlobalInitGenerator::Result
GlobalInitGenerator::VisitCharacterLiteral(const CharacterLiteral *expr) {
  unsigned value = expr->getValue();
  return Result::fromInt(
      value,
      irgen->typeConverter.VisitQualType(expr->getType(), expr->getBeginLoc())
          .cast<ir::IntT>(),
      expr->getSourceRange());
}

GlobalInitGenerator::Result
GlobalInitGenerator::VisitParenExpr(const ParenExpr *expr) {
  return Visit(expr->getSubExpr());
}

GlobalInitGenerator::Result
GlobalInitGenerator::VisitCastExpr(const CastExpr *expr) {
  auto subInit = Visit(expr->getSubExpr());
  assert(subInit && "Sub-expression in cast is invalid");

  auto srcType = irgen->typeConverter.VisitQualType(
      expr->getSubExpr()->getType(), expr->getSubExpr()->getBeginLoc());
  assert(srcType && "Source type conversion failed in cast");
  auto destType = irgen->typeConverter.VisitQualType(
      expr->getType(), llvm::isa<ImplicitCastExpr>(expr)
                           ? expr->getBeginLoc()
                           : llvm::cast<CStyleCastExpr>(expr)
                                 ->getTypeInfoAsWritten()
                                 ->getTypeLoc()
                                 .getBeginLoc());
  assert(destType && "Destination type conversion failed in cast");

  if (srcType == destType)
    return subInit;

  if (srcType.isa<ir::FloatT>() && destType.isa<ir::FloatT>()) {
    ir::FloatT srcFloatT = srcType.cast<ir::FloatT>();
    ir::FloatT destFloatT = destType.cast<ir::FloatT>();
    auto srcValue = subInit.getAPFloat();

    const auto &destSemantics = destFloatT.isF32()
                                    ? llvm::APFloat::IEEEsingle()
                                    : llvm::APFloat::IEEEdouble();

    bool losesInfo = false;
    srcValue.convert(destSemantics, llvm::APFloat::rmNearestTiesToEven,
                     &losesInfo);
    /// TODO: handle losesInfo?
    return {srcValue, expr->getSourceRange()};
  } else if (srcType.isa<ir::FloatT>() && !destType.isa<ir::FloatT>()) {
    auto srcValue = subInit.getAPFloat();
    assert(destType.isa<ir::IntT>() && "Integer type expected");
    auto intType = destType.cast<ir::IntT>();

    bool isExact = true;
    llvm::APSInt intValue;
    srcValue.convertToInteger(intValue, llvm::APFloat::rmTowardZero, &isExact);

    /// TODO: handle isExact?
    return {intValue, expr->getSourceRange()};
  } else if (!srcType.isa<ir::FloatT>() && destType.isa<ir::FloatT>()) {
    assert(srcType.isa<ir::IntT>() && "Integer type expected");
    auto srcValue = subInit.getAPSInt();
    auto floatType = destType.cast<ir::FloatT>();

    llvm::APFloat destValue(floatType.isF32() ? llvm::APFloat::IEEEsingle()
                                              : llvm::APFloat::IEEEdouble());
    auto status = destValue.convertFromAPInt(
        srcValue, srcValue.isSigned(), llvm::APFloat::rmNearestTiesToEven);
    assert(status == llvm::APFloat::opOK);
    (void)status;
    return {destValue, expr->getSourceRange()};
  } else {
    assert(srcType.isa<ir::IntT>() && destType.isa<ir::IntT>() &&
           "Integer type expected");
    auto srcValue = subInit.getAPSInt();
    assert(destType.isa<ir::IntT>() && "Integer type expected");
    auto destIntT = destType.cast<ir::IntT>();

    llvm::APSInt destValue = srcValue.extOrTrunc(destIntT.getBitWidth());
    destValue.setIsSigned(destIntT.isSigned());
    return {destValue, expr->getSourceRange()};
  }
}

GlobalInitGenerator::Result
GlobalInitGenerator::VisitInitListExpr(const InitListExpr *expr) {
  std::vector<std::unique_ptr<Result>> inits;
  inits.reserve(expr->getNumInits());
  for (const Expr *initExpr : expr->inits()) {
    auto init = Visit(initExpr);
    assert(init && "Initializer expression generation failed");
    inits.emplace_back(std::make_unique<Result>(std::move(init)));
  }

  return Result(std::move(inits), expr->getSourceRange());
}

GlobalInitGenerator::Result
GlobalInitGenerator::VisitConditionalOperator(const ConditionalOperator *expr) {
  auto condInit = Visit(expr->getCond());
  assert(condInit && "Condition expression generation failed");
  assert(condInit.isAPSInt() && "Condition must be an integer");

  auto condValue = condInit.getAPSInt();
  if (condValue.isZero()) {
    auto elseInit = Visit(expr->getFalseExpr());
    assert(elseInit && "Else expression generation failed");
    return elseInit;
  } else {
    auto thenInit = Visit(expr->getTrueExpr());
    assert(thenInit && "Then expression generation failed");
    return thenInit;
  }
}

#define ARITH_IMPL(operator, name)                                             \
  GlobalInitGenerator::Result GlobalInitGenerator::VisitBin##name##Op(         \
      const BinaryOperator *expr) {                                            \
    auto lhsInit = Visit(expr->getLHS());                                      \
    assert(lhsInit && "LHS expression generation failed");                     \
    auto rhsInit = Visit(expr->getRHS());                                      \
    assert(rhsInit && "RHS expression generation failed");                     \
    assert(lhsInit.isAPSInt() == rhsInit.isAPSInt() &&                         \
           "LHS and RHS must be of the same type");                            \
    if (lhsInit.isAPSInt()) {                                                  \
      auto lhsValue = lhsInit.getAPSInt();                                     \
      auto rhsValue = rhsInit.getAPSInt();                                     \
      auto resultValue = lhsValue operator rhsValue;                           \
      return {resultValue, expr->getSourceRange()};                            \
    } else {                                                                   \
      auto lhsValue = lhsInit.getAPFloat();                                    \
      auto rhsValue = rhsInit.getAPFloat();                                    \
      auto resultValue = lhsValue operator rhsValue;                           \
      return {resultValue, expr->getSourceRange()};                            \
    }                                                                          \
  }

ARITH_IMPL(*, Mul)
ARITH_IMPL(/, Div)
ARITH_IMPL(+, Add)
ARITH_IMPL(-, Sub)

#define INT_OP_IMPL(operator, name)                                            \
  GlobalInitGenerator::Result GlobalInitGenerator::VisitBin##name##Op(         \
      const BinaryOperator *expr) {                                            \
    auto lhsInit = Visit(expr->getLHS());                                      \
    assert(lhsInit && "LHS expression generation failed");                     \
    auto rhsInit = Visit(expr->getRHS());                                      \
    assert(rhsInit && "RHS expression generation failed");                     \
    assert(lhsInit.isAPSInt() == rhsInit.isAPSInt() &&                         \
           "LHS and RHS must be of the same type");                            \
    assert(lhsInit.isAPSInt() && "Bitwise operator only supports integers");   \
    auto lhsValue = lhsInit.getAPSInt();                                       \
    auto rhsValue = rhsInit.getAPSInt();                                       \
    auto resultValue = lhsValue operator rhsValue;                             \
    return {resultValue, expr->getSourceRange()};                              \
  }

INT_OP_IMPL(%, Rem)
INT_OP_IMPL(&, And)
INT_OP_IMPL(^, Xor)
INT_OP_IMPL(|, Or)

#define INT_SHIFT_IMPL(operator, name)                                         \
  GlobalInitGenerator::Result GlobalInitGenerator::VisitBin##name##Op(         \
      const BinaryOperator *expr) {                                            \
    auto lhsInit = Visit(expr->getLHS());                                      \
    assert(lhsInit && "LHS expression generation failed");                     \
    auto rhsInit = Visit(expr->getRHS());                                      \
    assert(rhsInit && "RHS expression generation failed");                     \
    assert(lhsInit.isAPSInt() == rhsInit.isAPSInt() &&                         \
           "LHS and RHS must be of the same type");                            \
    assert(lhsInit.isAPSInt() && "Shift operator only supports integers");     \
    auto lhsValue = lhsInit.getAPSInt();                                       \
    auto rhsValue = rhsInit.getAPSInt();                                       \
    assert(!rhsValue.isNegative() && "Negative shift count in global init");   \
    auto resultValue = lhsValue operator rhsValue.getZExtValue();              \
    return {resultValue, expr->getSourceRange()};                              \
  }

INT_SHIFT_IMPL(<<, Shl)
INT_SHIFT_IMPL(>>, Shr)

#define CMP_IMPL(operator, name)                                               \
  GlobalInitGenerator::Result GlobalInitGenerator::VisitBin##name##Op(         \
      const BinaryOperator *expr) {                                            \
    auto lhsInit = Visit(expr->getLHS());                                      \
    assert(lhsInit && "LHS expression generation failed");                     \
    auto rhsInit = Visit(expr->getRHS());                                      \
    assert(rhsInit && "RHS expression generation failed");                     \
    assert(lhsInit.isAPSInt() == rhsInit.isAPSInt() &&                         \
           "LHS and RHS must be of the same type");                            \
    if (lhsInit.isAPSInt()) {                                                  \
      auto lhsValue = lhsInit.getAPSInt();                                     \
      auto rhsValue = rhsInit.getAPSInt();                                     \
      bool resultValue = lhsValue operator rhsValue;                           \
      return Result::fromInt(                                                  \
          resultValue,                                                         \
          irgen->typeConverter                                                 \
              .VisitQualType(expr->getType(), expr->getBeginLoc())             \
              .cast<ir::IntT>(),                                               \
          expr->getSourceRange());                                             \
    } else {                                                                   \
      auto lhsValue = lhsInit.getAPFloat();                                    \
      auto rhsValue = rhsInit.getAPFloat();                                    \
      bool resultValue = lhsValue operator rhsValue;                           \
      return Result::fromInt(                                                  \
          resultValue,                                                         \
          irgen->typeConverter                                                 \
              .VisitQualType(expr->getType(), expr->getBeginLoc())             \
              .cast<ir::IntT>(),                                               \
          expr->getSourceRange());                                             \
    }                                                                          \
  }

CMP_IMPL(<, LT)
CMP_IMPL(>, GT)
CMP_IMPL(<=, LE)
CMP_IMPL(>=, GE)
CMP_IMPL(==, EQ)
CMP_IMPL(!=, NE)

#undef CMP_IMPL
#undef INT_SHIFT_IMPL
#undef INT_OP_IMPL
#undef ARITH_IMPL

GlobalInitGenerator::Result
GlobalInitGenerator::VisitBinLAndOp(const BinaryOperator *expr) {
  auto lhsInit = Visit(expr->getLHS());
  assert(lhsInit && "LHS expression generation failed");
  assert(lhsInit.isAPSInt() && "LHS must be an integer");
  auto lhsValue = lhsInit.getAPSInt();
  auto resultT =
      irgen->typeConverter.VisitQualType(expr->getType(), expr->getBeginLoc())
          .cast<ir::IntT>();
  if (lhsValue.isZero())
    return {lhsValue, !resultT.isSigned(), expr->getSourceRange()};

  auto rhsInit = Visit(expr->getRHS());
  assert(rhsInit && "RHS expression generation failed");
  assert(rhsInit.isAPSInt() && "RHS must be an integer");
  auto rhsValue = rhsInit.getAPSInt();
  if (rhsValue.isZero())
    return {rhsValue, !resultT.isSigned(), expr->getSourceRange()};

  return Result::fromInt(
      true,
      irgen->typeConverter.VisitQualType(expr->getType(), expr->getBeginLoc())
          .cast<ir::IntT>(),
      expr->getSourceRange());
}
GlobalInitGenerator::Result
GlobalInitGenerator::VisitBinLOrOp(const BinaryOperator *expr) {
  auto lhsInit = Visit(expr->getLHS());
  assert(lhsInit && "LHS expression generation failed");
  assert(lhsInit.isAPSInt() && "LHS must be an integer");
  auto lhsValue = lhsInit.getAPSInt();
  if (!lhsValue.isZero())
    return Result::fromInt(
        true,
        irgen->typeConverter.VisitQualType(expr->getType(), expr->getBeginLoc())
            .cast<ir::IntT>(),
        expr->getSourceRange());

  auto rhsInit = Visit(expr->getRHS());
  assert(rhsInit && "RHS expression generation failed");
  assert(rhsInit.isAPSInt() && "RHS must be an integer");
  auto rhsValue = rhsInit.getAPSInt();
  if (!rhsValue.isZero())
    return Result::fromInt(
        true,
        irgen->typeConverter.VisitQualType(expr->getType(), expr->getBeginLoc())
            .cast<ir::IntT>(),
        expr->getSourceRange());
  return Result::fromInt(
      false,
      irgen->typeConverter.VisitQualType(expr->getType(), expr->getBeginLoc())
          .cast<ir::IntT>(),
      expr->getSourceRange());
}

GlobalInitGenerator::Result
GlobalInitGenerator::VisitBinCommaOp(const BinaryOperator *expr) {
  auto rhsInit = Visit(expr->getRHS());
  assert(rhsInit && "RHS expression generation failed");
  return rhsInit;
}

GlobalInitGenerator::Result
GlobalInitGenerator::VisitUnPlusOp(const UnaryOperator *expr) {
  return Visit(expr->getSubExpr());
}
GlobalInitGenerator::Result
GlobalInitGenerator::VisitUnMinusOp(const UnaryOperator *expr) {
  auto subInit = Visit(expr->getSubExpr());
  assert(subInit && "Sub-expression in unary minus is invalid");

  if (subInit.isAPSInt()) {
    auto value = subInit.getAPSInt();
    value = -value;
    return {value, expr->getSourceRange()};
  } else {
    auto value = subInit.getAPFloat();
    value = -value;
    return {value, expr->getSourceRange()};
  }
}
GlobalInitGenerator::Result
GlobalInitGenerator::VisitUnNotOp(const UnaryOperator *expr) {
  auto subInit = Visit(expr->getSubExpr());
  assert(subInit && "Sub-expression in bitwise not is invalid");
  assert(subInit.isAPSInt() && "Bitwise not only supports integers");

  auto value = subInit.getAPSInt();
  value = ~value;
  return {value, expr->getSourceRange()};
}
GlobalInitGenerator::Result
GlobalInitGenerator::VisitUnLNotOp(const UnaryOperator *expr) {
  auto subInit = Visit(expr->getSubExpr());
  assert(subInit && "Sub-expression in logical not is invalid");
  assert(subInit.isAPSInt() && "Logical not only supports integers");

  auto value = subInit.getAPSInt();
  bool resultValue = value.isZero();
  return Result::fromInt(
      resultValue,
      irgen->typeConverter.VisitQualType(expr->getType(), expr->getBeginLoc())
          .cast<ir::IntT>(),
      expr->getSourceRange());
}

ir::InitializerAttr
GlobalInitGenerator::Result::toInitializerAttr(IRGenerator *irgen) {
  if (isAPSInt()) {
    llvm::SmallVector<char> buffer;
    getAPSInt().toString(buffer, 10);
    llvm::StringRef str(buffer.data(), buffer.size());

    ir::ASTInteger::Suffix suffix = getAPSInt().getBitWidth() <= 32
                                        ? ir::ASTInteger::Suffix::Int
                                        : ir::ASTInteger::Suffix::Long_L;
    return ir::ASTInteger::get(irgen->ctx, irgen->convertRange(range),
                               ir::ASTInteger::IntegerBase::Decimal, str,
                               suffix);
  } else if (isAPFloat()) {
    llvm::SmallVector<char> buffer;
    getAPFloat().toString(buffer);
    llvm::StringRef str(buffer.data(), buffer.size());
    ir::ASTFloat::Suffix suffix =
        &getAPFloat().getSemantics() == &llvm::APFloat::IEEEsingle()
            ? ir::ASTFloat::Suffix::Float_f
            : ir::ASTFloat::Suffix::Double;
    return ir::ASTFloat::get(irgen->ctx, irgen->convertRange(range), str,
                             suffix);
  } else if (isValueList()) {
    llvm::SmallVector<ir::InitializerAttr> elems;
    for (const auto &elem : getValueList())
      elems.emplace_back(elem->toInitializerAttr(irgen));

    return ir::ASTInitializerList::get(irgen->ctx, irgen->convertRange(range),
                                       elems);
  } else {
    llvm_unreachable("Invalid global initializer result");
  }
}

ir::Value ExprEvaluator::Visit(const Expr *E) {
  switch (E->getStmtClass()) {
  case Stmt::BinaryOperatorClass:
    return VisitBinaryOperator(llvm::cast<BinaryOperator>(E));
  case Stmt::UnaryOperatorClass:
    return VisitUnaryOperator(llvm::cast<UnaryOperator>(E));
  case Stmt::UnaryExprOrTypeTraitExprClass:
    return VisitUnaryExprOrTypeTraitExpr(
        llvm::cast<UnaryExprOrTypeTraitExpr>(E));
  case Stmt::IntegerLiteralClass:
    return VisitIntegerLiteral(llvm::cast<IntegerLiteral>(E));
  case Stmt::FloatingLiteralClass:
    return VisitFloatingLiteral(llvm::cast<FloatingLiteral>(E));
  case Stmt::CharacterLiteralClass:
    return VisitCharacterLiteral(llvm::cast<CharacterLiteral>(E));
  case Stmt::ParenExprClass:
    return VisitParenExpr(llvm::cast<ParenExpr>(E));
  case Stmt::ArraySubscriptExprClass:
    return VisitArraySubscriptExpr(llvm::cast<ArraySubscriptExpr>(E));
  case Stmt::ImplicitCastExprClass:
    return VisitCastExpr(llvm::cast<ImplicitCastExpr>(E));
  case Stmt::CStyleCastExprClass:
    return VisitCastExpr(llvm::cast<CStyleCastExpr>(E));
  case Stmt::MemberExprClass:
    return VisitMemberExpr(llvm::cast<MemberExpr>(E));
  case Stmt::ConditionalOperatorClass:
    return VisitConditionalOperator(llvm::cast<ConditionalOperator>(E));
  case Stmt::CallExprClass:
    return VisitCallExpr(llvm::cast<CallExpr>(E));
  case Stmt::DeclRefExprClass:
    return VisitDeclRefExpr(llvm::cast<DeclRefExpr>(E));

  default:
    llvm_unreachable("Unsupported expression in IR generation");
  }
}

ir::Value ExprEvaluator::VisitBinaryOperator(const BinaryOperator *expr) {
  switch (expr->getOpcode()) {
  case clang::BO_Mul:
    return VisitBinMulOp(expr);
  case clang::BO_Div:
    return VisitBinDivOp(expr);
  case clang::BO_Rem:
    return VisitBinRemOp(expr);
  case clang::BO_Add:
    return VisitBinAddOp(expr);
  case clang::BO_Sub:
    return VisitBinSubOp(expr);
  case clang::BO_Shl:
    return VisitBinShlOp(expr);
  case clang::BO_Shr:
    return VisitBinShrOp(expr);
  case clang::BO_LT:
    return VisitBinLTOp(expr);
  case clang::BO_GT:
    return VisitBinGTOp(expr);
  case clang::BO_LE:
    return VisitBinLEOp(expr);
  case clang::BO_GE:
    return VisitBinGEOp(expr);
  case clang::BO_EQ:
    return VisitBinEQOp(expr);
  case clang::BO_NE:
    return VisitBinNEOp(expr);
  case clang::BO_And:
    return VisitBinAndOp(expr);
  case clang::BO_Xor:
    return VisitBinXorOp(expr);
  case clang::BO_Or:
    return VisitBinOrOp(expr);
  case clang::BO_LAnd:
    return VisitBinLAndOp(expr);
  case clang::BO_LOr:
    return VisitBinLOrOp(expr);
  case clang::BO_Assign:
    return VisitBinAssignOp(expr);
  case clang::BO_MulAssign:
    return VisitBinMulAssignOp(expr);
  case clang::BO_DivAssign:
    return VisitBinDivAssignOp(expr);
  case clang::BO_RemAssign:
    return VisitBinRemAssignOp(expr);
  case clang::BO_AddAssign:
    return VisitBinAddAssignOp(expr);
  case clang::BO_SubAssign:
    return VisitBinSubAssignOp(expr);
  case clang::BO_ShlAssign:
    return VisitBinShlAssignOp(expr);
  case clang::BO_ShrAssign:
    return VisitBinShrAssignOp(expr);
  case clang::BO_AndAssign:
    return VisitBinAndAssignOp(expr);
  case clang::BO_XorAssign:
    return VisitBinXorAssignOp(expr);
  case clang::BO_OrAssign:
    return VisitBinOrAssignOp(expr);
  case clang::BO_Comma:
    return VisitBinCommaOp(expr);
  default:
    llvm_unreachable("Unsupported binary operator in IR generation");
  }
}

ir::Value ExprEvaluator::VisitUnaryOperator(const UnaryOperator *expr) {
  switch (expr->getOpcode()) {
  case clang::UO_Plus:
    return VisitUnPlusOp(expr);
  case clang::UO_Minus:
    return VisitUnMinusOp(expr);
  case clang::UO_Not:
    return VisitUnNotOp(expr);
  case clang::UO_LNot:
    return VisitUnLNotOp(expr);
  case clang::UO_PreInc:
    return VisitUnPreIncOp(expr);
  case clang::UO_PreDec:
    return VisitUnPreDecOp(expr);
  case clang::UO_PostInc:
    return VisitUnPostIncOp(expr);
  case clang::UO_PostDec:
    return VisitUnPostDecOp(expr);
  case clang::UO_AddrOf:
    return VisitUnAddrOfOp(expr);
  case clang::UO_Deref:
    return VisitUnDerefOp(expr);
  default:
    llvm_unreachable("Unsupported unary operator in IR generation");
  }
}

#define BINARY_IMPL(kind, name)                                                \
  ir::Value ExprEvaluator::VisitBin##name##Op(const BinaryOperator *expr) {    \
    ir::Value lhsV = Visit(expr->getLHS());                                    \
    assert(lhsV && "LHS expression generation failed");                        \
    ir::Value rhsV = Visit(expr->getRHS());                                    \
    assert(rhsV && "RHS expression generation failed");                        \
    assert(lhsV.getType() == rhsV.getType() &&                                 \
           "LHS and RHS must be of the same type");                            \
    return builder.create<ir::inst::Binary>(                                   \
        irgen->convertRange(expr->getSourceRange()), lhsV, rhsV,               \
        ir::inst::Binary::kind);                                               \
  }

BINARY_IMPL(Mul, Mul)
BINARY_IMPL(Add, Add)
BINARY_IMPL(Sub, Sub)
BINARY_IMPL(Div, Div)
BINARY_IMPL(Mod, Rem)
BINARY_IMPL(Shl, Shl)
BINARY_IMPL(Shr, Shr)

BINARY_IMPL(Lt, LT)
BINARY_IMPL(Gt, GT)
BINARY_IMPL(Le, LE)
BINARY_IMPL(Ge, GE)
BINARY_IMPL(Eq, EQ)
BINARY_IMPL(Ne, NE)

BINARY_IMPL(BitAnd, And)
BINARY_IMPL(BitXor, Xor)
BINARY_IMPL(BitOr, Or)

#undef BINARY_IMPL

ir::Value ExprEvaluator::VisitBinLAndOp(const BinaryOperator *expr) {
  ir::Value lhsV = Visit(expr->getLHS());
  assert(lhsV && "LHS expression generation failed");
  assert(lhsV.getType().isa<ir::IntT>() && "LHS must be an integer");

  ir::IntT lhsT = lhsV.getType().cast<ir::IntT>();
  ir::Value memory;
  {
    ir::IRBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(
        irgen->currentFunctionData->function->getAllocationBlock());

    auto boolT = ir::IntT::get(irgen->ctx, 1, false);
    memory = builder.create<ir::inst::LocalVariable>(
        irgen->convertRange(expr->getSourceRange()),
        ir::PointerT::get(irgen->ctx, boolT));
  }

  ir::Block *lhsTrueBlock = irgen->createNewBlock();
  ir::Block *lhsFalseBlock = irgen->createNewBlock();
  ir::Block *endBlock = irgen->createNewBlock();

  ir::JumpArgState lhsTrueJArg(lhsTrueBlock), lhsFalseJArg(lhsFalseBlock),
      endJArg(endBlock);

  if (!lhsT.isBoolean()) {
    auto zero = getIntegerValue(0, lhsT);
    lhsV = builder.create<ir::inst::Binary>(
        irgen->convertRange(expr->getLHS()->getSourceRange()), lhsV, zero,
        ir::inst::Binary::Ne);
  }

  builder.create<ir::inst::Branch>(
      irgen->convertRange(expr->getLHS()->getEndLoc()), lhsV, lhsTrueJArg,
      lhsFalseJArg);

  builder.setInsertionPoint(lhsTrueBlock);
  ir::Value rhsV = Visit(expr->getRHS());
  assert(rhsV && "RHS expression generation failed");
  assert(rhsV.getType().isa<ir::IntT>() && "RHS must be an integer");

  ir::IntT rhsT = rhsV.getType().cast<ir::IntT>();
  if (!rhsT.isBoolean()) {
    auto zero = getIntegerValue(0, rhsT);
    rhsV = builder.create<ir::inst::Binary>(
        irgen->convertRange(expr->getRHS()->getSourceRange()), rhsV, zero,
        ir::inst::Binary::Ne);
  }

  builder.create<ir::inst::Store>(
      irgen->convertRange(expr->getRHS()->getEndLoc()), rhsV, memory);

  builder.create<ir::inst::Jump>(
      irgen->convertRange(expr->getRHS()->getEndLoc()), endJArg);

  builder.setInsertionPoint(lhsFalseBlock);
  auto boolZero = getIntegerValue(0, ir::IntT::get(irgen->ctx, 1, false));
  builder.create<ir::inst::Store>(
      irgen->convertRange(expr->getLHS()->getEndLoc()), boolZero, memory);

  builder.create<ir::inst::Jump>(
      irgen->convertRange(expr->getLHS()->getEndLoc()), endJArg);

  builder.setInsertionPoint(endBlock);
  return builder.create<ir::inst::Load>(irgen->convertRange(expr->getEndLoc()),
                                        memory);
}

ir::Value ExprEvaluator::VisitBinLOrOp(const BinaryOperator *expr) {
  ir::Value lhsV = Visit(expr->getLHS());
  assert(lhsV && "LHS expression generation failed");
  assert(lhsV.getType().isa<ir::IntT>() && "LHS must be an integer");
  ir::IntT lhsT = lhsV.getType().cast<ir::IntT>();
  ir::Value memory;
  {
    ir::IRBuilder::InsertionGuard guard(builder);
    auto boolT = ir::IntT::get(irgen->ctx, 1, false);

    builder.setInsertionPoint(
        irgen->currentFunctionData->function->getAllocationBlock());

    memory = builder.create<ir::inst::LocalVariable>(
        irgen->convertRange(expr->getSourceRange()),
        ir::PointerT::get(irgen->ctx, boolT));
  }

  ir::Block *lhsTrueBlock = irgen->createNewBlock();
  ir::Block *lhsFalseBlock = irgen->createNewBlock();
  ir::Block *endBlock = irgen->createNewBlock();

  ir::JumpArgState lhsTrueJArg(lhsTrueBlock), lhsFalseJArg(lhsFalseBlock),
      endJArg(endBlock);

  if (!lhsT.isBoolean()) {
    auto zero = getIntegerValue(0, lhsT);
    lhsV = builder.create<ir::inst::Binary>(
        irgen->convertRange(expr->getLHS()->getSourceRange()), lhsV, zero,
        ir::inst::Binary::Ne);
  }

  builder.create<ir::inst::Branch>(
      irgen->convertRange(expr->getLHS()->getEndLoc()), lhsV, lhsTrueBlock,
      lhsFalseJArg);

  builder.setInsertionPoint(lhsTrueBlock);
  auto boolOne = getIntegerValue(1, ir::IntT::get(irgen->ctx, 1, false));
  builder.create<ir::inst::Store>(
      irgen->convertRange(expr->getLHS()->getEndLoc()), boolOne, memory);
  builder.create<ir::inst::Jump>(
      irgen->convertRange(expr->getLHS()->getEndLoc()), endJArg);

  builder.setInsertionPoint(lhsFalseBlock);
  ir::Value rhsV = Visit(expr->getRHS());
  assert(rhsV && "RHS expression generation failed");
  assert(rhsV.getType().isa<ir::IntT>() && "RHS must be an integer");
  ir::IntT rhsT = rhsV.getType().cast<ir::IntT>();
  if (!rhsT.isBoolean()) {
    auto zero = getIntegerValue(0, rhsT);
    rhsV = builder.create<ir::inst::Binary>(
        irgen->convertRange(expr->getRHS()->getSourceRange()), rhsV, zero,
        ir::inst::Binary::Ne);
  }

  builder.create<ir::inst::Store>(
      irgen->convertRange(expr->getRHS()->getEndLoc()), rhsV, memory);
  builder.create<ir::inst::Jump>(
      irgen->convertRange(expr->getRHS()->getEndLoc()), endJArg);

  builder.setInsertionPoint(endBlock);
  return builder.create<ir::inst::Load>(irgen->convertRange(expr->getEndLoc()),
                                        memory);
}

ir::Value ExprEvaluator::VisitBinCommaOp(const BinaryOperator *expr) {
  Visit(expr->getLHS());
  return Visit(expr->getRHS());
}

ir::Value ExprEvaluator::VisitBinAssignOp(const BinaryOperator *expr) {
  ir::Value lhsV = Visit(expr->getLHS());
  assert(lhsV && "LHS expression generation failed");
  assert(expr->getLHS()->isLValue() && "LHS of assignment must be an lvalue");
  assert(lhsV.getType().isa<ir::PointerT>() &&
         "LHS of assignment must be a pointer");

  ir::Type pointeeType = lhsV.getType().cast<ir::PointerT>().getPointeeType();
  ir::Value rhsV = Visit(expr->getRHS());
  assert(rhsV && "RHS expression generation failed");
  assert(rhsV.getType() == pointeeType &&
         "RHS type must match the LHS pointee type");

  builder.create<ir::inst::Store>(irgen->convertRange(expr->getSourceRange()),
                                  rhsV, lhsV);
  return rhsV;
}

#define BIN_ASSIGN_IMPL(kind, name)                                            \
  ir::Value ExprEvaluator::VisitBin##name##AssignOp(                           \
      const BinaryOperator *expr) {                                            \
    ir::Value lhsV = Visit(expr->getLHS());                                    \
    assert(lhsV && "LHS expression generation failed");                        \
    assert(expr->getLHS()->isLValue() &&                                       \
           "LHS of assignment must be an lvalue");                             \
    assert(lhsV.getType().isa<ir::PointerT>() &&                               \
           "LHS of assignment must be a pointer");                             \
                                                                               \
    ir::Type pointeeType =                                                     \
        lhsV.getType().cast<ir::PointerT>().getPointeeType();                  \
                                                                               \
    ir::Value loadedLhsV = builder.create<ir::inst::Load>(                     \
        irgen->convertRange(expr->getLHS()->getEndLoc()), lhsV);               \
    ir::Value rhsV = Visit(expr->getRHS());                                    \
    assert(rhsV && "RHS expression generation failed");                        \
    auto exprT = irgen->typeConverter.VisitQualType(expr->getType(),           \
                                                    expr->getBeginLoc());      \
                                                                               \
    if (loadedLhsV.getType() != exprT) {                                       \
      loadedLhsV = builder.create<ir::inst::TypeCast>(                         \
          irgen->convertRange(expr->getRHS()->getEndLoc()), loadedLhsV,        \
          exprT);                                                              \
    }                                                                          \
                                                                               \
    if (rhsV.getType() != exprT) {                                             \
      rhsV = builder.create<ir::inst::TypeCast>(                               \
          irgen->convertRange(expr->getRHS()->getEndLoc()), rhsV, exprT);      \
    }                                                                          \
                                                                               \
    ir::Value resultV = builder.create<ir::inst::Binary>(                      \
        irgen->convertRange(expr->getSourceRange()), loadedLhsV, rhsV,         \
        ir::inst::Binary::kind);                                               \
                                                                               \
    if (resultV.getType() != pointeeType) {                                    \
      resultV = builder.create<ir::inst::TypeCast>(                            \
          irgen->convertRange(expr->getRHS()->getEndLoc()), resultV,           \
          pointeeType);                                                        \
    }                                                                          \
    builder.create<ir::inst::Store>(                                           \
        irgen->convertRange(expr->getSourceRange()), resultV, lhsV);           \
    return resultV;                                                            \
  }

BIN_ASSIGN_IMPL(Add, Add)
BIN_ASSIGN_IMPL(Sub, Sub)
BIN_ASSIGN_IMPL(Mod, Rem)
BIN_ASSIGN_IMPL(Div, Div)
BIN_ASSIGN_IMPL(Mul, Mul)
BIN_ASSIGN_IMPL(Shl, Shl)
BIN_ASSIGN_IMPL(Shr, Shr)
BIN_ASSIGN_IMPL(BitAnd, And)
BIN_ASSIGN_IMPL(BitXor, Xor)
BIN_ASSIGN_IMPL(BitOr, Or)

#undef BIN_ASSIGN_IMPL

ir::Value ExprEvaluator::VisitUnPlusOp(const UnaryOperator *expr) {
  auto subV = Visit(expr->getSubExpr());
  assert(subV && "Sub-expression in unary plus is invalid");
  return builder.create<ir::inst::Unary>(
      irgen->convertRange(expr->getSourceRange()), subV, ir::inst::Unary::Plus);
}

ir::Value ExprEvaluator::VisitUnMinusOp(const UnaryOperator *expr) {
  auto subV = Visit(expr->getSubExpr());
  assert(subV && "Sub-expression in unary minus is invalid");
  return builder.create<ir::inst::Unary>(
      irgen->convertRange(expr->getSourceRange()), subV,
      ir::inst::Unary::Minus);
}

ir::Value ExprEvaluator::VisitUnNotOp(const UnaryOperator *expr) {
  auto subV = Visit(expr->getSubExpr());
  assert(subV && "Sub-expression in bitwise not is invalid");
  assert(subV.getType().isa<ir::IntT>() &&
         "Bitwise not only supports integers");
  return builder.create<ir::inst::Unary>(
      irgen->convertRange(expr->getSourceRange()), subV,
      ir::inst::Unary::Negate);
}

ir::Value ExprEvaluator::VisitUnLNotOp(const UnaryOperator *expr) {
  auto subV = Visit(expr->getSubExpr());
  assert(subV && "Sub-expression in bitwise not is invalid");
  assert(subV.getType().isa<ir::IntT>() &&
         "Bitwise not only supports integers");
  auto subT = subV.getType().cast<ir::IntT>();
  if (!subT.isBoolean()) {
    auto zero = getIntegerValue(0, subT);
    subV = builder.create<ir::inst::Binary>(
        irgen->convertRange(expr->getSubExpr()->getSourceRange()), subV, zero,
        ir::inst::Binary::Ne);
  }
  auto boolOne = getIntegerValue(1, ir::IntT::get(irgen->ctx, 1, false));
  return builder.create<ir::inst::Binary>(
      irgen->convertRange(expr->getSourceRange()), subV, boolOne,
      ir::inst::Binary::BitXor);
}

ir::Value ExprEvaluator::VisitUnPreIncOp(const UnaryOperator *expr) {
  assert(expr->getSubExpr()->isLValue() &&
         "Operand of pre-increment must be an lvalue");
  ir::Value subV = Visit(expr->getSubExpr());
  assert(subV && "Sub-expression in pre-increment is invalid");
  assert(subV.getType().isa<ir::PointerT>() &&
         "Operand of pre-increment must be a pointer");

  ir::Type pointeeType = subV.getType().cast<ir::PointerT>().getPointeeType();
  assert(pointeeType.isa<ir::IntT>() &&
         "Pre-increment only supports integer types");

  ir::IntT intT = pointeeType.cast<ir::IntT>();
  auto one = getIntegerValue(1, intT);
  auto loadedV = builder.create<ir::inst::Load>(
      irgen->convertRange(expr->getSubExpr()->getSourceRange()), subV);
  auto resultV = builder.create<ir::inst::Binary>(
      irgen->convertRange(expr->getSourceRange()), loadedV, one,
      ir::inst::Binary::Add);
  builder.create<ir::inst::Store>(irgen->convertRange(expr->getSourceRange()),
                                  resultV, subV);
  return resultV;
}

ir::Value ExprEvaluator::VisitUnPreDecOp(const UnaryOperator *expr) {
  assert(expr->getSubExpr()->isLValue() &&
         "Operand of pre-decrement must be an lvalue");
  ir::Value subV = Visit(expr->getSubExpr());
  assert(subV && "Sub-expression in pre-decrement is invalid");
  assert(subV.getType().isa<ir::PointerT>() &&
         "Operand of pre-decrement must be a pointer");

  ir::Type pointeeType = subV.getType().cast<ir::PointerT>().getPointeeType();
  assert(pointeeType.isa<ir::IntT>() &&
         "Pre-decrement only supports integer types");
  ir::IntT intT = pointeeType.cast<ir::IntT>();
  auto one = getIntegerValue(1, intT);
  auto loadedV = builder.create<ir::inst::Load>(
      irgen->convertRange(expr->getSubExpr()->getSourceRange()), subV);
  auto resultV = builder.create<ir::inst::Binary>(
      irgen->convertRange(expr->getSourceRange()), loadedV, one,
      ir::inst::Binary::Sub);
  builder.create<ir::inst::Store>(irgen->convertRange(expr->getSourceRange()),
                                  resultV, subV);
  return resultV;
}

ir::Value ExprEvaluator::VisitUnPostIncOp(const UnaryOperator *expr) {
  assert(expr->getSubExpr()->isLValue() &&
         "Operand of post-increment must be an lvalue");
  ir::Value subV = Visit(expr->getSubExpr());
  assert(subV && "Sub-expression in post-increment is invalid");
  assert(subV.getType().isa<ir::PointerT>() &&
         "Operand of post-increment must be a pointer");

  ir::Type pointeeType = subV.getType().cast<ir::PointerT>().getPointeeType();
  assert(pointeeType.isa<ir::IntT>() &&
         "Post-increment only supports integer types");
  ir::IntT intT = pointeeType.cast<ir::IntT>();
  auto one = getIntegerValue(1, intT);
  auto loadedV = builder.create<ir::inst::Load>(
      irgen->convertRange(expr->getSubExpr()->getSourceRange()), subV);
  auto resultV = builder.create<ir::inst::Binary>(
      irgen->convertRange(expr->getSourceRange()), loadedV, one,
      ir::inst::Binary::Add);
  builder.create<ir::inst::Store>(irgen->convertRange(expr->getSourceRange()),
                                  resultV, subV);
  return loadedV;
}

ir::Value ExprEvaluator::VisitUnPostDecOp(const UnaryOperator *expr) {
  assert(expr->getSubExpr()->isLValue() &&
         "Operand of post-decrement must be an lvalue");
  ir::Value subV = Visit(expr->getSubExpr());
  assert(subV && "Sub-expression in post-decrement is invalid");
  assert(subV.getType().isa<ir::PointerT>() &&
         "Operand of post-decrement must be a pointer");

  ir::Type pointeeType = subV.getType().cast<ir::PointerT>().getPointeeType();
  assert(pointeeType.isa<ir::IntT>() &&
         "Post-decrement only supports integer types");
  ir::IntT intT = pointeeType.cast<ir::IntT>();
  auto one = getIntegerValue(1, intT);
  auto loadedV = builder.create<ir::inst::Load>(
      irgen->convertRange(expr->getSubExpr()->getSourceRange()), subV);
  auto resultV = builder.create<ir::inst::Binary>(
      irgen->convertRange(expr->getSourceRange()), loadedV, one,
      ir::inst::Binary::Sub);
  builder.create<ir::inst::Store>(irgen->convertRange(expr->getSourceRange()),
                                  resultV, subV);
  return loadedV;
}

ir::Value ExprEvaluator::VisitUnAddrOfOp(const UnaryOperator *expr) {
  assert(expr->getSubExpr()->isLValue() &&
         "Operand of address-of must be an lvalue");
  ir::Value subV = Visit(expr->getSubExpr());
  assert(subV && "Sub-expression in address-of is invalid");
  assert(subV.getType().isa<ir::PointerT>() &&
         "Operand of address-of must be a pointer");
  return subV;
}

ir::Value ExprEvaluator::VisitUnDerefOp(const UnaryOperator *expr) {
  ir::Value subV = Visit(expr->getSubExpr());
  assert(subV && "Sub-expression in dereference is invalid");
  assert(subV.getType().isa<ir::PointerT>() &&
         "Operand of dereference must be a pointer");
  return builder.create<ir::inst::Load>(
      irgen->convertRange(expr->getSourceRange()), subV);
}

ir::Value ExprEvaluator::VisitUnaryExprOrTypeTraitExpr(
    const UnaryExprOrTypeTraitExpr *expr) {
  size_t value = 0;
  switch (expr->getKind()) {
  case clang::UETT_SizeOf: {
    ir::Type type;
    if (expr->isArgumentType()) {
      type = irgen->typeConverter.VisitQualType(
          expr->getArgumentType(),
          expr->getArgumentTypeInfo()->getTypeLoc().getBeginLoc());
    } else {
      type = irgen->typeConverter.VisitQualType(
          expr->getArgumentExpr()->getType(),
          expr->getArgumentExpr()->getBeginLoc());
    }
    assert(type && "Type conversion failed in sizeof");
    value = type.getSizeAndAlign(irgen->recordDeclMgr.getStructSizeMap()).first;
  }
  case clang::UETT_AlignOf: {
    ir::Type type;
    if (expr->isArgumentType()) {
      type = irgen->typeConverter.VisitQualType(
          expr->getArgumentType(),
          expr->getArgumentTypeInfo()->getTypeLoc().getBeginLoc());
    } else {
      type = irgen->typeConverter.VisitQualType(
          expr->getArgumentExpr()->getType(),
          expr->getArgumentExpr()->getBeginLoc());
    }
    assert(type && "Type conversion failed in alignof");
    value =
        type.getSizeAndAlign(irgen->recordDeclMgr.getStructSizeMap()).second;
  }
  default:
    llvm_unreachable("Unsupported unary expr or type trait in IR generation");
  }

  ir::IRBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(irgen->ir->getConstantBlock());
  auto type =
      irgen->typeConverter.VisitQualType(expr->getType(), expr->getBeginLoc());
  assert(type && "Type conversion failed in sizeof/alignof");
  auto intT = type.cast<ir::IntT>();

  return builder.create<ir::inst::Constant>(
      irgen->convertRange(expr->getSourceRange()),
      ir::ConstantIntAttr::get(irgen->ctx, value, intT.getBitWidth(),
                               intT.isSigned()));
}

ir::Value ExprEvaluator::VisitIntegerLiteral(const IntegerLiteral *expr) {
  auto type =
      irgen->typeConverter.VisitQualType(expr->getType(), expr->getBeginLoc());
  assert(type && "Type conversion failed in integer literal");
  auto intT = type.cast<ir::IntT>();

  llvm::APInt value = expr->getValue();
  ir::IRBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(irgen->ir->getConstantBlock());

  return builder.create<ir::inst::Constant>(
      irgen->convertRange(expr->getSourceRange()),
      ir::ConstantIntAttr::get(irgen->ctx,
                               intT.isSigned() ? value.getSExtValue()
                                               : value.getZExtValue(),
                               intT.getBitWidth(), intT.isSigned()));
}

ir::Value ExprEvaluator::VisitFloatingLiteral(const FloatingLiteral *expr) {
  ir::IRBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(irgen->ir->getConstantBlock());
  auto value = expr->getValue();
  return builder.create<ir::inst::Constant>(
      irgen->convertRange(expr->getSourceRange()),
      ir::ConstantFloatAttr::get(irgen->ctx, value));
}

ir::Value ExprEvaluator::VisitCharacterLiteral(const CharacterLiteral *expr) {
  auto type =
      irgen->typeConverter.VisitQualType(expr->getType(), expr->getBeginLoc());
  assert(type && "Type conversion failed in character literal");
  auto intT = type.cast<ir::IntT>();

  auto value = expr->getValue();
  ir::IRBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(irgen->ir->getConstantBlock());
  return builder.create<ir::inst::Constant>(
      irgen->convertRange(expr->getSourceRange()),
      ir::ConstantIntAttr::get(irgen->ctx, value, intT.getBitWidth(),
                               intT.isSigned()));
}

ir::Value ExprEvaluator::VisitParenExpr(const ParenExpr *expr) {
  return Visit(expr->getSubExpr());
}

ir::Value
ExprEvaluator::VisitArraySubscriptExpr(const ArraySubscriptExpr *expr) {
  assert(expr->getBase()->isLValue() &&
         "Base of array subscript must be an lvalue");
  ir::Value baseV = Visit(expr->getBase());
  assert(baseV && "Base expression generation failed");
  assert(baseV.getType().isa<ir::PointerT>() &&
         "Base of array subscript must be a pointer");

  ir::Type baseT = baseV.getType();
  ir::ArrayT arrayT =
      baseT.cast<ir::PointerT>().getPointeeType().cast<ir::ArrayT>();
  ir::Type elemT = arrayT.getElementType();
  auto [elemSize, elemAlign] =
      elemT.getSizeAndAlign(irgen->recordDeclMgr.getStructSizeMap());
  if (elemSize < elemAlign)
    elemSize = elemAlign;

  ir::Value sizeV;
  {
    ir::IRBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(irgen->ir->getConstantBlock());
    sizeV = builder.create<ir::inst::Constant>(
        irgen->convertRange(expr->getSourceRange()),
        ir::ConstantIntAttr::get(irgen->ctx, elemSize,
                                 irgen->ctx->getArchitectureBitSize(), true));
  }

  ir::Value idxV = Visit(expr->getIdx());
  assert(idxV && "Index expression generation failed");
  assert(idxV.getType().isa<ir::IntT>() && "Index must be an integer");
  ir::IntT idxIntT = idxV.getType().cast<ir::IntT>();
  if (idxIntT.getBitWidth() < irgen->ctx->getArchitectureBitSize()) {
    idxV = builder.create<ir::inst::TypeCast>(
        irgen->convertRange(expr->getIdx()->getSourceRange()), idxV,
        ir::IntT::get(irgen->ctx, irgen->ctx->getArchitectureBitSize(), true));
    idxIntT = idxV.getType().cast<ir::IntT>();
  }

  auto mulV = builder.create<ir::inst::Binary>(
      irgen->convertRange(expr->getSourceRange()), idxV, sizeV,
      ir::inst::Binary::Mul);

  auto gepV = builder.create<ir::inst::Gep>(
      irgen->convertRange(expr->getSourceRange()), baseV, mulV,
      ir::PointerT::get(irgen->ctx, elemT));

  return gepV;
}

ir::Value ExprEvaluator::VisitCastExpr(const CastExpr *expr) {
  ir::Value subV = Visit(expr->getSubExpr());
  assert(subV && "Sub-expression generation failed");

  auto destT =
      irgen->typeConverter.VisitQualType(expr->getType(), expr->getBeginLoc());
  assert(destT && "Type conversion failed in cast expression");

  switch (expr->getCastKind()) {
  case clang::CK_LValueToRValue: {
    assert(subV.getType().isa<ir::PointerT>() &&
           "LValue to RValue cast source must be a pointer");
    ir::Type pointeeType = subV.getType().cast<ir::PointerT>().getPointeeType();
    assert(pointeeType == destT &&
           "LValue to RValue cast source and destination type must match");
    return builder.create<ir::inst::Load>(
        irgen->convertRange(expr->getEndLoc()), subV);
  }
  case clang::CK_NoOp: {
    return subV;
  }
  case clang::CK_ArrayToPointerDecay: {
    assert(subV.getType().isa<ir::PointerT>() &&
           "Array to pointer decay source must be a pointer");
    ir::Type pointeeType = subV.getType().cast<ir::PointerT>().getPointeeType();
    assert(pointeeType.isa<ir::ArrayT>() &&
           "Array to pointer decay source must be a pointer to array");
    ir::Type elemT = pointeeType.cast<ir::ArrayT>().getElementType();
    ir::Type decayT = ir::PointerT::get(irgen->ctx, elemT);

    ir::Value zero = getIntegerValue(
        0,
        ir::IntT::get(irgen->ctx, irgen->ctx->getArchitectureBitSize(), true));
    return builder.create<ir::inst::Gep>(
        irgen->convertRange(expr->getSourceRange()), subV, zero, decayT);
  }
  case clang::CK_BitCast: {
    ir::Type srcT = subV.getType();
    assert(srcT.isa<ir::PointerT>() && destT.isa<ir::PointerT>() &&
           "Bitcast only supports pointer types");
    ir::Value zero = getIntegerValue(
        0,
        ir::IntT::get(irgen->ctx, irgen->ctx->getArchitectureBitSize(), true));
    return builder.create<ir::inst::Gep>(
        irgen->convertRange(expr->getSourceRange()), subV, zero, destT);
  }
  case clang::CK_FunctionToPointerDecay: {
    // Do nothing
    return subV;
  }
  case clang::CK_IntegralCast:
  case clang::CK_IntegralToBoolean:
  case clang::CK_IntegralToFloating:
  case clang::CK_NullToPointer:
  case clang::CK_IntegralToPointer:
  case clang::CK_PointerToIntegral:
  case clang::CK_PointerToBoolean:
  case clang::CK_ToVoid:
  case clang::CK_FloatingToIntegral:
  case clang::CK_FloatingToBoolean:
  case clang::CK_BooleanToSignedIntegral:
  case clang::CK_FloatingCast: {
    ir::Type srcT = subV.getType();
    return builder.create<ir::inst::TypeCast>(
        irgen->convertRange(expr->getSourceRange()), subV, destT);
  }
  default:
    llvm_unreachable("Unsupported cast kind in IR generation");
  }
}

ir::Value ExprEvaluator::VisitMemberExpr(const MemberExpr *expr) {
  assert((expr->isArrow() || expr->getBase()->isLValue()) &&
         "Base of member expression must be an lvalue");

  ir::Value baseV = Visit(expr->getBase());
  assert(baseV && "Base expression generation failed");
  assert(baseV.getType().isa<ir::PointerT>() &&
         "Base of member expression must be a pointer");

  ir::Type baseT = baseV.getType();
  ir::NameStruct structT =
      baseT.cast<ir::PointerT>().getPointeeType().cast<ir::NameStruct>();

  expr->getMemberDecl();

  const auto &[size, align, offsets] =
      irgen->recordDeclMgr.getStructSizeMap().at(structT.getName());
  const auto &fieldTypes =
      irgen->recordDeclMgr.getStructFieldsMap().at(structT.getName());
  const auto &fieldIdxMap =
      irgen->recordDeclMgr.getFieldIndexMap(structT.getName());
  auto memberIdx = fieldIdxMap.at(expr->getMemberDecl()->getName());
  auto memberOffset = offsets[memberIdx];
  ir::Value offsetV = getIntegerValue(
      memberOffset,
      ir::IntT::get(irgen->ctx, irgen->ctx->getArchitectureBitSize(), true));

  auto memberT = fieldTypes[memberIdx];
  return builder.create<ir::inst::Gep>(
      irgen->convertRange(expr->getSourceRange()), baseV, offsetV,
      ir::PointerT::get(irgen->ctx, memberT));
}

ir::Value
ExprEvaluator::VisitConditionalOperator(const ConditionalOperator *expr) {
  ir::Value condV = Visit(expr->getCond());
  assert(condV && "Condition expression generation failed");
  assert(condV.getType().isa<ir::IntT>() && "Condition must be an integer");

  ir::IntT intT = condV.getType().cast<ir::IntT>();
  ir::Value zero = getIntegerValue(0, intT);
  ir::Value memory;
  {
    ir::IRBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(
        irgen->currentFunctionData->function->getAllocationBlock());

    auto exprT = irgen->typeConverter.VisitQualType(expr->getType(),
                                                    expr->getBeginLoc());
    assert(exprT && "Type conversion failed in conditional operator");
    auto boolT = ir::IntT::get(irgen->ctx, 1, false);
    memory = builder.create<ir::inst::LocalVariable>(
        irgen->convertRange(expr->getSourceRange()),
        ir::PointerT::get(irgen->ctx, exprT));
  }

  ir::Block *trueBlock = irgen->createNewBlock();
  ir::Block *falseBlock = irgen->createNewBlock();
  ir::Block *endBlock = irgen->createNewBlock();

  ir::JumpArgState trueJArg(trueBlock), falseJArg(falseBlock),
      endJArg(endBlock);

  if (!intT.isBoolean()) {
    condV = builder.create<ir::inst::Binary>(
        irgen->convertRange(expr->getCond()->getSourceRange()), condV, zero,
        ir::inst::Binary::Ne);
  }

  builder.create<ir::inst::Branch>(
      irgen->convertRange(expr->getCond()->getEndLoc()), condV, trueJArg,
      falseJArg);

  builder.setInsertionPoint(trueBlock);
  ir::Value trueV = Visit(expr->getTrueExpr());
  assert(trueV && "True expression generation failed");
  builder.create<ir::inst::Store>(
      irgen->convertRange(expr->getTrueExpr()->getEndLoc()), trueV, memory);

  builder.create<ir::inst::Jump>(
      irgen->convertRange(expr->getTrueExpr()->getEndLoc()), endJArg);

  builder.setInsertionPoint(falseBlock);
  ir::Value falseV = Visit(expr->getFalseExpr());
  assert(falseV && "False expression generation failed");
  builder.create<ir::inst::Store>(
      irgen->convertRange(expr->getFalseExpr()->getEndLoc()), falseV, memory);
  builder.create<ir::inst::Jump>(
      irgen->convertRange(expr->getFalseExpr()->getEndLoc()), endJArg);
  builder.setInsertionPoint(endBlock);

  assert(trueV.getType() == falseV.getType() &&
         "True and false expression must be of the same type");
  return builder.create<ir::inst::Load>(irgen->convertRange(expr->getEndLoc()),
                                        memory);
}

ir::Value ExprEvaluator::VisitCallExpr(const CallExpr *expr) {
  ir::Value calleeV = Visit(expr->getCallee());
  assert(calleeV && "Callee expression generation failed");
  assert(calleeV.getType().isa<ir::PointerT>() &&
         "Callee must be a pointer to function");
  assert(calleeV.getType()
             .cast<ir::PointerT>()
             .getPointeeType()
             .isa<ir::FunctionT>() &&
         "Callee must be a pointer to function");
  ir::FunctionT funcT = calleeV.getType()
                            .cast<ir::PointerT>()
                            .getPointeeType()
                            .cast<ir::FunctionT>();
  assert(funcT.getArgTypes().size() == expr->getNumArgs() &&
         "Number of arguments must match number of parameters");

  llvm::SmallVector<ir::Value> argVs;
  argVs.reserve(expr->getNumArgs());
  for (size_t i = 0; i < expr->getNumArgs(); ++i) {
    ir::Value argV = Visit(expr->getArg(i));
    assert(argV && "Argument expression generation failed");
    assert(argV.getType() == funcT.getArgTypes()[i] &&
           "Argument type must match parameter type");
    argVs.emplace_back(argV);
  }

  auto callV = builder.create<ir::inst::Call>(
      irgen->convertRange(expr->getSourceRange()), calleeV, argVs);
  assert(callV->getResultSize() == 1 && "Call must have one result");
  return callV->getResult(0);
}

ir::Value ExprEvaluator::VisitDeclRefExpr(const clang::DeclRefExpr *expr) {
  auto nameInfo = expr->getNameInfo().getName().getAsIdentifierInfo();
  assert(nameInfo && "DeclRefExpr must have an identifier");

  auto name = nameInfo->getName();
  auto value = irgen->env.lookup(name);
  assert(value && "Identifier not found in environment");
  return value;
}

ir::Value ExprEvaluator::getIntegerValue(int64_t value, ir::IntT intT) {
  ir::IRBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(irgen->ir->getConstantBlock());
  return builder.create<ir::inst::Constant>(
      {}, ir::ConstantIntAttr::get(irgen->ctx, value, intT.getBitWidth(),
                                   intT.isSigned()));
}

} // namespace kecc::c
