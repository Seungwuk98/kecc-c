#include "kecc/c/IRGenerator.h"
#include "kecc/ir/IRAttributes.h"
#include "kecc/ir/IRBuilder.h"
#include "kecc/ir/IRInstructions.h"
#include "clang/AST/Decl.h"

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

void IRGenerator::VisitVarDecl(const clang::VarDecl *D) {
  auto type =
      typeConverter.VisitQualType(D->getType(), D->getTypeSpecStartLoc());
  assert(type && "Variable type conversion failed");

  ir::IRBuilder::InsertionGuard guard(builder);

  if (env.isGlobalScope()) {
    builder.setInsertionPoint(ir->getGlobalBlock());
    ir::InitializerAttr init;
    if (D->hasInit()) {
      GlobalInitGenerator initGen(this);
      init = initGen.Visit(D->getInit());
      assert(init && "Global variable initializer generation failed");
    }
    builder.create<ir::inst::GlobalVariableDefinition>(
        convertRange(D->getSourceRange()), D->getName(), type, init);

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
        convertRange(D->getSourceRange()), name, ir::PointerT::get(ctx, type));
    env.insert(name, localVar);
    /// TODO: handle local variable initializer
  }
}

} // namespace kecc::c
