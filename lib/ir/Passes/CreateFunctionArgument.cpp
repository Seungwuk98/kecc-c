#include "kecc/ir/IRAttributes.h"
#include "kecc/ir/IRBuilder.h"
#include "kecc/ir/IRInstructions.h"
#include "kecc/ir/IRTransforms.h"
#include "kecc/ir/IRTypes.h"

namespace kecc::ir {

PassResult CreateFunctionArgument::run(Module *module, Function *func) {
  if (!func->hasDefinition())
    return PassResult::skip();

  auto funcType = func->getFunctionType().cast<FunctionT>();
  auto argTypes = funcType.getArgTypes();
  if (argTypes.empty())
    return PassResult::skip();

  Block *entryBlock = func->getEntryBlock();
  auto entryPreds = module->getPredecessors(entryBlock);

  IRBuilder builder(func->getContext());
  builder.setInsertionPointStart(entryBlock);
  if (entryPreds.empty()) {
    // phi instructions are not needed if there are no predecessors
    // change phi arguments to function arguments

    size_t phiIdx = 0;

    llvm::SmallVector<std::pair<Phi, inst::FunctionArgument>> replacements;
    replacements.reserve(argTypes.size());
    for (auto I = entryBlock->phiBegin(); phiIdx < argTypes.size();
         ++I, ++phiIdx) {
      auto phi = (*I)->getDefiningInst<Phi>();
      assert(phi && "Phi instruction expected");
      auto argType = argTypes[phiIdx];
      assert(argType == phi.getType() &&
             "Phi argument type must match function argument type");
      auto arg =
          builder.create<inst::FunctionArgument>(phi.getRange(), argType);
      replacements.emplace_back(phi, arg);
    }

    for (const auto &[phi, arg] : replacements)
      module->replaceInst(phi.getStorage(), arg.getStorage());
  } else {
    // create new entry block
    auto entryId = entryBlock->getId();

    while (func->getBlockById(entryId))
      entryId++;

    Block *newBlock = new Block(entryId);
    newBlock->setParentFunction(func);
    func->begin().getNode()->prev->insertNext(newBlock);

    builder.setInsertionPointStart(newBlock);

    llvm::SmallVector<Value> jumpArgValues;
    jumpArgValues.reserve(argTypes.size());

    size_t phiIdx = 0;
    auto I = entryBlock->phiBegin();
    for (; phiIdx < argTypes.size(); ++I, ++phiIdx) {
      auto phi = (*I)->getDefiningInst<Phi>();
      assert(phi && "Phi instruction expected");
      auto argType = argTypes[phiIdx];
      assert(argType == phi.getType() &&
             "Phi argument type must match function argument type");
      auto arg =
          builder.create<inst::FunctionArgument>(phi.getRange(), argType);
      jumpArgValues.emplace_back(arg);
    }

    if ((*I)->getDefiningInst<Phi>()) {
      IRBuilder::InsertionGuard guard(builder);
      builder.setInsertionPoint(module->getIR()->getConstantBlock());

      for (; (*I)->getDefiningInst<Phi>(); ++I) {
        auto phi = (*I)->getDefiningInst<Phi>();
        assert(phi && "Phi instruction expected");
        auto argType = phi.getType();
        auto undef = builder.create<inst::Constant>(
            {}, ConstantUndefAttr::get(builder.getContext(), argType));
        jumpArgValues.emplace_back(undef);
      }
    }

    JumpArgState jumpArg;
    jumpArg.setArgs(jumpArgValues);
    jumpArg.setBlock(entryBlock);
    builder.create<inst::Jump>({}, jumpArg);

    func->setEntryBlock(newBlock->getId());
    module->addBlockRelation(newBlock, entryBlock);
  }

  return PassResult::success();
}

} // namespace kecc::ir
