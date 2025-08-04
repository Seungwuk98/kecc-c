#include "kecc/ir/Context.h"
#include "kecc/ir/IRAnalyses.h"
#include "kecc/ir/IRInstructions.h"
#include "kecc/ir/IRTransforms.h"
#include "kecc/ir/IRTypes.h"
#include "kecc/ir/Instruction.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/Support/MemAlloc.h"
#include "llvm/Support/TrailingObjects.h"
#include <optional>

namespace kecc::ir {

using GVNIndex = size_t;

using InstFindKey = std::tuple<TypeID, llvm::ArrayRef<GVNIndex>,
                               llvm::ArrayRef<Attribute>, Type>;
using InstHashKey = std::tuple<TypeID, llvm::SmallVector<GVNIndex>,
                               llvm::SmallVector<Attribute>, Type>;

struct InstDenseMapInfo {
  static InstHashKey getEmptyKey() {
    return {
        TypeID::get<void>(), {}, {}, llvm::DenseMapInfo<Type>::getEmptyKey()};
  }

  static InstHashKey getTombstoneKey() {
    return {TypeID::get<void>(),
            {},
            {},
            llvm::DenseMapInfo<Type>::getTombstoneKey()};
  }

  static llvm::hash_code getHashValue(const InstHashKey &key) {
    return llvm::DenseMapInfo<InstFindKey>::getHashValue(
        {std::get<0>(key), std::get<1>(key), std::get<2>(key),
         std::get<3>(key)});
  }

  static llvm::hash_code getHashValue(const InstFindKey &key) {
    return llvm::DenseMapInfo<InstFindKey>::getHashValue(key);
  }

  static bool isEqual(const InstHashKey &lhs, const InstHashKey &rhs) {
    return std::get<0>(lhs) == std::get<0>(rhs) &&
           std::get<1>(lhs) == std::get<1>(rhs) &&
           std::get<2>(lhs) == std::get<2>(rhs) &&
           std::get<3>(lhs) == std::get<3>(rhs);
  }

  static bool isEqual(const InstFindKey &lhs, const InstHashKey &rhs) {
    return std::get<0>(lhs) == std::get<0>(rhs) &&
           std::get<1>(lhs) == llvm::ArrayRef(std::get<1>(rhs)) &&
           std::get<2>(lhs) == llvm::ArrayRef(std::get<2>(rhs)) &&
           std::get<3>(lhs) == std::get<3>(rhs);
  }
};

class GVNTable {
public:
  GVNTable() = default;
  GVNTable(const GVNTable &) = default;
  GVNTable &operator=(const GVNTable &) = default;

  GVNIndex getGvnE(Value value) const {
    assert(valueToGVNEMap.contains(value) &&
           "Value must be registered before getting its GVN expression");
    return valueToGVNEMap.at(value);
  }

  GVNIndex hasGvnE(Value value) const { return valueToGVNEMap.contains(value); }

  GVNIndex createOrGetGvnE(Value value) {
    auto it = valueToGVNEMap.find(value);
    if (it != valueToGVNEMap.end())
      return it->second;
    auto newE = id++;
    valueToGVNEMap.try_emplace(value, newE);
    return newE;
  }

  GVNIndex registerGvnE(Value value) {
    assert(!valueToGVNEMap.contains(value) &&
           "Value should not be registered before getting its GVN expression");
    assert(!value.getDefiningInst<inst::Unresolved>() &&
           "unresolved values should not be registered by this function");

    if (value.getDefiningInst<inst::Call>() || value.getDefiningInst<Phi>() ||
        value.getDefiningInst<inst::Constant>() ||
        value.getDefiningInst<inst::Load>() ||
        value.getDefiningInst<inst::Store>() ||
        value.getDefiningInst<inst::LocalVariable>())
      // Call instructions could have side effects, so we don't cache the
      // expression. Only track expression.
      // Phi instructions are classified by type and their parent block.
      // GVNE doesn't track parent block
      return createOrGetGvnE(value);

    InstructionStorage *instStorage = value.getInstruction();

    auto typeID = instStorage->getAbstractInstruction()->getId();
    llvm::SmallVector<GVNIndex> exprs;
    for (Value operand : instStorage->getOperands())
      exprs.emplace_back(getGvnE(operand));

    llvm::ArrayRef<Attribute> attrs = instStorage->getAttributes();
    InstFindKey key{typeID, exprs, attrs, value.getType()};
    GVNIndex gvnE;
    if (auto it = instToGVNEMap.find_as(key); it != instToGVNEMap.end()) {
      gvnE = it->second; // Found existing GVN expression
    } else {
      gvnE = id++;
      InstHashKey hashKey{typeID, exprs, attrs, value.getType()};
      instToGVNEMap.try_emplace(hashKey, gvnE);
    }
    valueToGVNEMap.try_emplace(value, gvnE);
    return gvnE;
  }

  Value findGVNE(Block *block, GVNIndex gvnE) const {
    auto it = gvnEMap.at(block).find(gvnE);
    if (it != gvnEMap.at(block).end()) {
      return it->second; // Return the value associated with the expression
    }
    return nullptr; // Expression not found in the block's set
  }

  // Inserts a value into the block's expression set.
  // If the value is already present, it returns the existing value and gvn
  // expression.
  // If the value is not present, it inserts the new value and returns a null
  // and gvn expression.
  std::pair<Value, GVNIndex> insertValue(Block *block, Value value) {
    auto gvnE = registerGvnE(value);
    auto &blockGVNEMap = gvnEMap[block];

    if (auto it = blockGVNEMap.find(gvnE); it != blockGVNEMap.end()) {
      // If the expression already exists, return the existing value.
      return {it->second, gvnE};
    }
    // If the expression was not found, insert the new value and return null
    blockGVNEMap.try_emplace(gvnE, value);
    return {nullptr, gvnE};
  }

  void setValueToGvnE(Value value, GVNIndex gvnE) {
    assert(value.getDefiningInst<Phi>() ||
           value.getDefiningInst<inst::Unresolved>() &&
               "Only Phi or Unresolved instructions can be set as GVN "
               "expressions");
    valueToGVNEMap[value] = gvnE;
  }

  void setGvnEWithValue(Block *block, GVNIndex gvnE, Value value) {
    assert(value.getDefiningInst<Phi>() ||
           value.getDefiningInst<inst::Unresolved>() &&
               "Only Phi or Unresolved instructions can be inserted as GVN "
               "expressions");
    auto &exprSet = gvnEMap[block];
    exprSet[gvnE] = value;
    valueToGVNEMap[value] = gvnE; // Store the expression with a new id
  }

  Value findValue(Block *block, GVNIndex gvnE) const {
    auto it = gvnEMap.find(block);
    if (it != gvnEMap.end()) {
      const auto &exprSet = it->second;
      auto exprIt = exprSet.find(gvnE);
      if (exprIt != exprSet.end()) {
        return exprIt
            ->second; // Return the value associated with the expression
      }
    }
    return nullptr; // Expression not found in the block's set
  }

  void insertBlockTable(Block *block, Block *idom) {
    auto &exprSet = gvnEMap[block];
    if (idom) {
      for (const auto &[idomExpr, idomValue] : gvnEMap[idom])
        exprSet.try_emplace(idomExpr, idomValue);
    }
  }

private:
  std::size_t id = 0;
  llvm::DenseMap<Value, GVNIndex> valueToGVNEMap;
  llvm::DenseMap<InstHashKey, GVNIndex, InstDenseMapInfo> instToGVNEMap;
  llvm::DenseMap<Block *, llvm::DenseMap<GVNIndex, Value>> gvnEMap;
};

class GVNImpl {
public:
  GVNImpl(Module *module, Function *function, const DominatorTree *domTree,
          llvm::ArrayRef<Block *> rpo, const GVNTable &constantTable)
      : module(module), function(function), domTree(domTree), rpo(rpo),
        constantTable(constantTable) {}

  PassResult run();

  Phi phiInsertion(inst::Unresolved unresolved, Block *currBlock,
                   llvm::ArrayRef<std::pair<Block *, Value>> predValues);

  void initPerFunction(Module *module, Function *function);

private:
  Module *module;
  Function *function;
  const DominatorTree *domTree;
  llvm::ArrayRef<Block *> rpo;
  GVNTable table;
  const GVNTable constantTable;
};

GVN::GVN() = default;
GVN::~GVN() = default;

PassResult GVN::run(Module *module, Function *function) { return impl->run(); }

void GVN::init(Module *module) {
  auto domAnalysis = module->getAnalysis<DominanceAnalysis>();
  if (!domAnalysis) {
    auto newDomAnalysis = DominanceAnalysis::create(module);
    module->insertAnalysis(std::move(newDomAnalysis));
    domAnalysis = module->getAnalysis<DominanceAnalysis>();
  }

  auto visitOrderAnalysis = module->getAnalysis<VisitOrderAnalysis>();
  if (!visitOrderAnalysis) {
    auto newVisitOrderAnalysis = VisitOrderAnalysis::create(module);
    module->insertAnalysis(std::move(newVisitOrderAnalysis));
    visitOrderAnalysis = module->getAnalysis<VisitOrderAnalysis>();
  }

  GVNTable table;
  for (InstructionStorage *inst : *module->getIR()->getConstantBlock()) {
    auto constant = inst->getDefiningInst<inst::Constant>();
    table.insertValue(module->getIR()->getConstantBlock(), constant);
  }

  GVNTable copiedTable = table;

  impl =
      std::make_unique<GVNImpl>(module, nullptr, nullptr, std::nullopt, table);
}

void GVN::exit(Module *module) { impl.reset(); }

void GVN::init(Module *module, Function *function) {
  impl->initPerFunction(module, function);
}

void GVNImpl::initPerFunction(Module *module, Function *function) {
  this->function = function;
  this->domTree =
      module->getAnalysis<DominanceAnalysis>()->getDominatorTree(function);
  this->rpo =
      module->getAnalysis<VisitOrderAnalysis>()->getReversePostOrder(function);
  this->table = this->constantTable;
  this->table.insertBlockTable(function->getEntryBlock(),
                               module->getIR()->getConstantBlock());
}

Phi GVNImpl::phiInsertion(
    inst::Unresolved unresolved, Block *currBlock,
    llvm::ArrayRef<std::pair<Block *, Value>> predValues) {
  auto it = currBlock->phiEnd();
  it--;
  IRBuilder builder(module->getContext());
  builder.setInsertionPoint(Block::InsertionPoint(currBlock, it));

  auto phi = builder.create<Phi>(unresolved.getRange(), unresolved.getType());

  for (auto [pred, predValue] : predValues) {
    module->replaceExit(
        pred->getExit(),
        [&](IRBuilder &builder, BlockExit oldExit) -> BlockExit {
          for (auto [idx, jumpArg] :
               llvm::enumerate(oldExit.getStorage()->getJumpArgs())) {
            if (jumpArg->getBlock() == currBlock) {
              auto state = jumpArg->getAsState();
              state.pushArg(predValue);
              oldExit.getStorage()->setJumpArg(idx, state);
            }
          }
          return oldExit;
        });
  }

  return phi;
}

PassResult GVNImpl::run() {
  bool changed = false;

  for (InstructionStorage *inst : *function->getAllocationBlock()) {
    auto alloc = inst->getDefiningInst<inst::LocalVariable>();
    table.insertValue(function->getAllocationBlock(), alloc);
  }

  table.insertBlockTable(function->getEntryBlock(),
                         function->getAllocationBlock());

  for (Block *block : rpo) {
    // Initialize the block's expression set with its immediate dominator's set
    Block *idom = domTree->getIdom(block);
    table.insertBlockTable(block, idom);

    llvm::DenseMap<Value, Value> replaceMap;
    llvm::DenseMap<inst::Unresolved, /* temp instruction before insertion phi */
                   llvm::SmallVector<std::pair<Block *, Value>> /* argument value with each
                                                predecessors */>
        phiInsertMap;

    auto preds = module->getPredecessors(block);

    size_t phiIndex = 0;
    for (auto I = block->phiBegin(), E = block->phiEnd(); I != E;
         ++I, ++phiIndex) {
      auto phi = (*I)->getDefiningInst<Phi>();
      if (!preds.empty()) {
        std::optional<GVNIndex> argV;
        bool same = true;
        for (Block *pred : preds) {
          auto predExit = pred->getExit();
          auto walkResult = predExit.walk([&](JumpArg *jumpArg) -> WalkResult {
            if (jumpArg->getBlock() == block) {
              Value arg = jumpArg->getArgs()[phiIndex];
              if (!table.hasGvnE(arg))
                return WalkResult::interrupt();
              GVNIndex argIndex = table.getGvnE(arg);
              if (!argV) {
                argV = argIndex;
              } else if (argIndex != *argV)
                return WalkResult::interrupt();
            }
            return WalkResult::advance();
          });
          if (walkResult.isInterrupt()) {
            same = false;
            break;
          }
        }

        if (same) {
          table.setValueToGvnE(phi, *argV);
          auto findV = table.findValue(block, *argV);
          assert(findV && "Value must be found in the block's expression set");
          replaceMap.try_emplace(phi.getResult(), findV);
        } else
          table.insertValue(block, phi);
      } else {
        table.insertValue(block, phi);
      }
    }

    for (auto I = block->tempBegin(), E = block->tempEnd(); I != E; ++I) {
      InstructionStorage *inst = *I;
      llvm::SmallVector<Value> results = inst->getResults();
      for (Value result : results) {
        auto [existingValue, gvnE] = table.insertValue(block, result);

        if (existingValue) {
          replaceMap.try_emplace(result, existingValue);
          continue;
        }

        if (!preds.empty()) {
          llvm::SmallVector<std::pair<Block *, Value>> predValues;
          predValues.reserve(preds.size());
          bool success = true;
          for (Block *pred : preds) {
            auto predValue = table.findValue(pred, gvnE);
            if (!predValue) {
              success = false;
              break;
            }
            predValues.emplace_back(pred, predValue);
          }

          if (success) {
            IRBuilder builder(module->getContext());
            builder.setInsertionPoint(function->getUnresolvedBlock());
            auto unresolved = builder.create<inst::Unresolved>(
                inst->getRange(), result.getType());
            table.setGvnEWithValue(block, gvnE, unresolved);
            replaceMap.try_emplace(result, unresolved);
            phiInsertMap.try_emplace(unresolved, std::move(predValues));
          }
        }
      }
    }

    for (auto [oldV, newV] : replaceMap) {
      while (replaceMap.contains(newV))
        newV = replaceMap.at(newV);

      oldV.replaceWith(newV);
      changed = true;
    }

    for (auto [unresolved, predValues] : phiInsertMap) {
      auto phi = phiInsertion(unresolved, block, predValues);
      unresolved.getResult().replaceWith(phi);
      changed = true;
      auto gvnE = table.getGvnE(unresolved);
      table.setGvnEWithValue(block, gvnE, phi);
    }
  }
  return changed ? PassResult::success() : PassResult::skip();
}

} // namespace kecc::ir
