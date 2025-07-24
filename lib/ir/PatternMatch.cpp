#include "kecc/ir/PatternMatch.h"
#include "kecc/ir/Attribute.h"
#include "kecc/ir/IRAnalyses.h"
#include "kecc/ir/Instruction.h"
#include "kecc/ir/JumpArg.h"
#include "kecc/ir/WalkSupport.h"
#include "kecc/utils/LogicalResult.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVectorExtras.h"

namespace kecc::ir {

TransactionInstState TransactionInstState::create(InstructionStorage *inst) {
  TransactionInstState state;
  state.inst = inst;
  state.range = inst->getRange();
  state.operands =
      llvm::map_to_vector(inst->getOperands(), [](Value v) { return v; });
  state.types = llvm::map_to_vector(
      inst->getResults(), [](Value v) -> std::pair<Type, std::string> {
        return {v.getType(), v.getValueName().str()};
      });
  state.attributes = llvm::SmallVector<Attribute>(inst->getAttributes());
  state.jumpArgs = llvm::map_to_vector(
      inst->getJumpArgs(),
      [](JumpArg *arg) -> JumpArgState { return arg->getAsState(); });
  state.parentBlock = inst->getParentBlock();
  state.abstractInst = inst->getAbstractInstruction();
  return state;
}

void TransactionInstState::rollback() {
  inst->setRange(range);
  inst->setOperands(operands);
  inst->setAttributes(attributes);
  inst->setJumpArgs(jumpArgs);
  inst->setParentBlock(parentBlock);
  inst->setAbstractInstruction(abstractInst);
}

TransactionFunctionState TransactionFunctionState::create(Function *function) {
  TransactionFunctionState state;
  state.function = function;
  state.name = function->getName().str();
  state.functionType = function->getFunctionType();
  state.entryBid = function->getEntryBlock()->getId();
  return state;
}

void TransactionFunctionState::rollback() {
  function->setName(name);
  function->setFunctionType(functionType);
  function->setEntryBlock(entryBid);
}

struct IRRewriterImpl {
  IRRewriter::State getState() {
    return {transactionStates.size(), createdInsts.size(), removedInsts.size(),
            functionActions.size(),   blockActions.size(), replaceMap.size()};
  }

  std::vector<TransactionInstState> transactionStates;
  std::vector<FunctionAction> functionActions;
  std::vector<InstructionStorage *> createdInsts;
  llvm::SetVector<InstructionStorage *> removedInsts;
  std::vector<BlockAction> blockActions;
  std::vector<std::pair<Value, Value>> replaceMap;
};

IRRewriter::IRRewriter(Module *module, IRContext *context)
    : IRBuilder(context), module(module), impl(new IRRewriterImpl) {}
IRRewriter::~IRRewriter() {}

void IRRewriter::notifyInstCreated(InstructionStorage *inst) {
  impl->createdInsts.emplace_back(inst);
}

void IRRewriter::removeInst(InstructionStorage *inst) {
  assert(inst && !isInstIgnored(inst) &&
         "Cannot remove an ignored instruction");

  impl->removedInsts.insert(inst);
}

void IRRewriter::notifyStartUpdate(InstructionStorage *inst) {
  assert(inst && !isInstIgnored(inst) &&
         "Cannot update an ignored instruction");
  impl->transactionStates.emplace_back(TransactionInstState::create(inst));
}

void IRRewriter::notifyFunctionCreated(Function *func) {
  assert(func && "Cannot create a null function");
  impl->functionActions.emplace_back(FunctionAction::created(func));
}

void IRRewriter::notifyStartUpdate(Function *func) {
  assert(func && "Cannot modify a null function");
  impl->functionActions.emplace_back(FunctionAction::modifyStart(func));
}

void IRRewriter::notifyBlockCreated(Block *block) {
  assert(block && "Cannot create a null block");
  impl->blockActions.emplace_back(BlockAction::created(block));
}

void IRRewriter::notifyBlockRemoved(Block *block) {
  assert(block && "Cannot remove a null block");
  impl->blockActions.emplace_back(BlockAction::removed(block));
}

IRRewriter::State IRRewriter::getCurrentState() const {
  return impl->getState();
}

void IRRewriter::replaceInst(InstructionStorage *inst,
                             llvm::ArrayRef<Value> values) {
  assert(inst && !isInstIgnored(inst) &&
         "Cannot replace an ignored instruction");

  llvm::SmallVector<Value> results = inst->getResults();
  assert(results.size() == values.size() &&
         "Number of results must match the number of replacement values");
  for (auto [from, to] : llvm::zip(results, values)) {
    if (from != to)
      replaceValue(from, to);
  }

  impl->removedInsts.insert(inst);
}

void IRRewriter::replaceValue(Value from, Value to) {
  impl->replaceMap.emplace_back(from, to);
}

void IRRewriter::resetToState(const State &state) {
  for (size_t i = state.transactionCount; i < impl->transactionStates.size();
       ++i) {
    impl->transactionStates[i].rollback();
  }
  impl->transactionStates.erase(impl->transactionStates.begin() +
                                    state.transactionCount,
                                impl->transactionStates.end());

  for (size_t i = state.createdInstsCount; i < impl->createdInsts.size(); ++i) {
    InstructionStorage *inst = impl->createdInsts[i];
    inst->dropReferences();
    inst->getParentBlock()->remove(inst);
  }
  impl->createdInsts.erase(impl->createdInsts.begin() + state.createdInstsCount,
                           impl->createdInsts.end());

  llvm::DenseSet<InstructionStorage *> removedSet;
  for (size_t i = state.removedInstsCount; i < impl->removedInsts.size(); ++i) {
    InstructionStorage *inst = impl->removedInsts[i];
    removedSet.insert(inst);
  }
  for (InstructionStorage *inst : removedSet)
    impl->removedInsts.remove(inst);

  for (size_t i = state.functionActionCount; i < impl->functionActions.size();
       ++i) {
    const FunctionAction &action = impl->functionActions[i];
    if (action.getKind() == FunctionAction::Kind::Create) {
      Function *function = action.getFunction();
      function->dropReferences();
      function->getParentIR()->erase(function->getName());
    } else if (action.getKind() == FunctionAction::Kind::Modified) {
      impl->functionActions[i].getTransactionState()->rollback();
    }
  }
  impl->functionActions.erase(impl->functionActions.begin() +
                                  state.functionActionCount,
                              impl->functionActions.end());

  for (size_t i = state.blockActionCount; i < impl->blockActions.size(); ++i) {
    const BlockAction &action = impl->blockActions[i];
    if (action.getKind() == BlockAction::Kind::Create) {
      Block *block = action.getBlock();
      block->dropReferences();
      block->getParentFunction()->eraseBlock(block->getId());
    }
  }
  impl->blockActions.erase(impl->blockActions.begin() + state.blockActionCount,
                           impl->blockActions.end());

  impl->replaceMap.erase(impl->replaceMap.begin() + state.replaceMapCount,
                         impl->replaceMap.end());
}

bool IRRewriter::isInstIgnored(InstructionStorage *inst) const {
  return impl->removedInsts.contains(inst);
}

void IRRewriter::discardRewrite() { resetToState({}); }
void IRRewriter::applyRewrite() {
  llvm::DenseMap<Value, Value> replaceMap;
  for (auto [from, to] : impl->replaceMap) {
    if (auto it = replaceMap.find(to); it != replaceMap.end()) {
      to = it->second;
    }
    from.replaceWith(to);
    replaceMap[from] = to; // Update the map to handle chained replacements
  }

  for (InstructionStorage *toRemove : impl->removedInsts) {
    toRemove->dropReferences();
    toRemove->getParentBlock()->remove(toRemove);
  }

  for (const auto &action : impl->functionActions) {
    if (action.getKind() == FunctionAction::Kind::Remove) {
      Function *function = action.getFunction();
      function->dropReferences();
      function->getParentIR()->erase(function->getName());
    }
  }

  for (const auto &action : impl->blockActions) {
    if (action.getKind() == BlockAction::Kind::Remove) {
      Block *block = action.getBlock();
      block->dropReferences();
      block->getParentFunction()->eraseBlock(block->getId());
    }
  }

  module->updatePredsAndSuccs();
}

std::pair<llvm::SmallVector<Value>, llvm::SmallVector<JumpArgState>>
IRRewriter::remapping(llvm::ArrayRef<Operand> operands,
                      llvm::ArrayRef<JumpArg *> jumpArgs) {

  llvm::DenseMap<Value, Value> mapping;

  for (auto [from, to] : impl->replaceMap) {
    auto it = mapping.find(to);
    if (it != mapping.end()) {
      to = it->second;
    }
    mapping[from] = to; // Update the mapping
  }

  llvm::SmallVector<Value> remappedOperands;
  remappedOperands.reserve(operands.size());
  llvm::SmallVector<JumpArgState> remappedJumpArgs;
  remappedJumpArgs.reserve(jumpArgs.size());

  for (Value operand : operands) {
    auto it = mapping.find(operand);
    if (it != mapping.end())
      remappedOperands.push_back(it->second);
    else
      remappedOperands.push_back(operand);
  }

  for (JumpArg *arg : jumpArgs) {
    auto state = arg->getAsState();
    for (auto [idx, value] : llvm::enumerate(state.getArgs())) {
      auto it = mapping.find(value);
      if (it != mapping.end())
        state.setArg(idx, it->second);
    }
    remappedJumpArgs.push_back(state);
  }

  return {remappedOperands, remappedJumpArgs};
}

OrderedPatternSet::OrderedPatternSet(const PatternSet &set) {
  for (const auto &pattern : set.getPatterns()) {
    const auto &patternId = pattern->getPatternId();
    if (patternId.isInstruction()) {
      auto instId = patternId.getId();
      pushInstPattern(instId, pattern.get());
    } else if (patternId.isInterface()) {
      auto interfaceId = patternId.getId();
      pushInterfacePattern(interfaceId, pattern.get());
    } else if (patternId.isTrait()) {
      auto blockId = patternId.getId();
      pushTraitPattern(blockId, pattern.get());
    } else
      pushGeneralPattern(pattern.get());
  }
}

void OrderedPatternSet::pushGeneralPattern(Pattern *pattern) {
  generalPatterns.emplace_back(pattern);
}
void OrderedPatternSet::pushInstPattern(TypeID id, Pattern *pattern) {
  instPatternMap[id].emplace_back(pattern);
}
void OrderedPatternSet::pushInterfacePattern(TypeID id, Pattern *pattern) {
  interfacePatternMap[id].emplace_back(pattern);
}
void OrderedPatternSet::pushTraitPattern(TypeID id, Pattern *pattern) {
  traitPatternMap[id].emplace_back(pattern);
}
llvm::SmallVector<Pattern *>
OrderedPatternSet::getAppliablePatterns(InstructionStorage *inst) const {
  auto instId = inst->getAbstractInstruction()->getId();
  llvm::SmallVector<Pattern *> patterns = instPatternMap.lookup(instId);

  for (const auto &[interfaceId, interfacePatterns] : interfacePatternMap) {
    if (inst->hasInterface(interfaceId)) {
      patterns.insert(patterns.end(), interfacePatterns.begin(),
                      interfacePatterns.end());
    }
  }

  for (const auto &[traitId, traitPatterns] : traitPatternMap) {
    if (inst->hasTrait(traitId)) {
      patterns.insert(patterns.end(), traitPatterns.begin(),
                      traitPatterns.end());
    }
  }

  patterns.insert(patterns.end(), generalPatterns.begin(),
                  generalPatterns.end());

  return patterns;
}

class PatternApplier {
public:
  PatternApplier(Module *module, const PatternSet &patternSet,
                 IRRewriter &rewriter)
      : module(module), pattern(patternSet), rewriter(rewriter) {}

  utils::LogicalResult apply();

  utils::LogicalResult convert(InstructionStorage *inst);

  bool canApply(InstructionStorage *inst, Pattern *pattern);

private:
  Module *module;
  OrderedPatternSet pattern;
  IRRewriter &rewriter;

  llvm::DenseSet<std::pair<InstructionStorage *, Pattern *>> appliedPatterns;
};

utils::LogicalResult applyPatternConversion(Module *module,
                                            const PatternSet &patterns) {
  IRRewriter rewriter(module, module->getContext());
  PatternApplier applier(module, patterns, rewriter);

  auto result = applier.apply();
  if (result.isError()) {
    rewriter.discardRewrite();
    return result;
  }
  rewriter.applyRewrite();
  return utils::LogicalResult::success();
}

utils::LogicalResult PatternApplier::apply() {
  auto *orderAnalysis = module->getAnalysis<ir::VisitOrderAnalysis>();
  if (!orderAnalysis) {
    auto analysis = ir::VisitOrderAnalysis::create(module);
    assert(analysis && "VisitOrderAnalysis should be created successfully");
    module->insertAnalysis(std::move(analysis));
    orderAnalysis = module->getAnalysis<ir::VisitOrderAnalysis>();
  }

  llvm::SmallVector<InstructionStorage *> toConvert;

  for (Function *function : *module->getIR()) {
    auto rpo = orderAnalysis->getReversePostOrder(function);
    for (Block *block : rpo) {
      block->walk([&](InstructionStorage *inst) -> WalkResult {
        toConvert.emplace_back(inst);
        return WalkResult::advance();
      });
    }
  }

  for (InstructionStorage *inst : toConvert) {
    auto result = convert(inst);
    if (result.isError())
      return result;
  }

  return utils::LogicalResult::success(); // success
}

bool PatternApplier::canApply(InstructionStorage *inst, Pattern *pattern) {
  if (appliedPatterns.insert({inst, pattern}).second) {
    return true;
  }
  return false;
}

utils::LogicalResult PatternApplier::convert(InstructionStorage *inst) {
  if (rewriter.isInstIgnored(inst))
    return utils::LogicalResult::failure();

  auto rewriterState = rewriter.getCurrentState();
  auto patterns = pattern.getAppliablePatterns(inst);

  if (patterns.empty())
    return utils::LogicalResult::failure(); // no patterns to apply

  llvm::stable_sort(patterns, [](const Pattern *a, const Pattern *b) {
    return a->getBenefit() > b->getBenefit();
  });

  bool matched = false;
  for (Pattern *pattern : patterns) {
    if (!canApply(inst, pattern))
      continue; // cannot apply this pattern

    IRBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointBeforeInst(inst);
    auto patternMatched = pattern->matchAndRewrite(rewriter, inst);
    if (!patternMatched.succeeded()) {
      rewriter.resetToState(rewriterState);
      if (patternMatched.isError())
        break;
      else
        continue;
    }

    matched = true;
    break;
  }

  if (!matched)
    /// no pattern matched, reset to the previous state
    return utils::LogicalResult::failure();

  for (size_t i = rewriterState.transactionCount,
              e = rewriter.getImpl()->transactionStates.size();
       i < e; ++i) {
    const TransactionInstState &state =
        rewriter.getImpl()->transactionStates[i];
    InstructionStorage *modified = state.getInst();
    auto convertResult =
        convert(modified); // recursively convert modified instructions
    if (convertResult.isError()) {
      rewriter.resetToState(rewriterState);
      return convertResult;
    }
  }

  for (size_t i = rewriterState.createdInstsCount,
              e = rewriter.getImpl()->createdInsts.size();
       i < e; ++i) {
    InstructionStorage *created = rewriter.getImpl()->createdInsts[i];
    auto convertResult =
        convert(created); // recursively convert created instructions

    if (convertResult.isError()) {
      rewriter.resetToState(rewriterState);
      return convertResult;
    }
  }

  return utils::LogicalResult::success(); // conversion succeeded
}
} // namespace kecc::ir
