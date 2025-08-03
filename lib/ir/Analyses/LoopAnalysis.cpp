#include "kecc/ir/Analysis.h"
#include "kecc/ir/IRAnalyses.h"
#include "kecc/ir/Type.h"

DEFINE_KECC_TYPE_ID(kecc::ir::LoopAnalysis)

namespace kecc::ir {

class LoopTree {
public:
  using LoopType = LoopAnalysis::LoopType;
  LoopTree(llvm::DenseMap<Block *, llvm::DenseSet<Block *>> loopTree,
           llvm::DenseMap<Block *, Block *> loopHeader,
           llvm::DenseMap<Block *, llvm::DenseSet<LoopType>> loopType)
      : loopTree(std::move(loopTree)), loopHeader(std::move(loopHeader)),
        loopType(std::move(loopType)) {}

  LoopTree(const LoopTree &other) = default;
  LoopTree &operator=(const LoopTree &other) = default;
  LoopTree(LoopTree &&other) = default;
  LoopTree &operator=(LoopTree &&other) = default;

  const llvm::DenseSet<LoopType> &getLoopTypes(Block *block) const;

  // Returns the loop type of the block.
  // Usually the program is reducible, so it returns NonHeader, Reducible, or
  // SelfLoop.
  LoopType getLoopType(Block *block) const;

  const llvm::DenseSet<Block *> &getLoopChildren(Block *block) const;
  Block *getLoopHeader(Block *block) const;

  void dump(llvm::raw_ostream &os) const;

private:
  llvm::DenseMap<Block *, llvm::DenseSet<Block *>> loopTree;
  llvm::DenseMap<Block *, Block *> loopHeader;
  llvm::DenseMap<Block *, llvm::DenseSet<LoopType>> loopType;
};

const llvm::DenseSet<LoopTree::LoopType> &
LoopTree::getLoopTypes(Block *block) const {
  return loopType.at(block);
}

LoopTree::LoopType LoopTree::getLoopType(Block *block) const {
  auto loopTypes = getLoopTypes(block);
  assert(llvm::all_equal(loopTypes) &&
         "Loop types for a block should be the same for all dummies");
  assert(*loopTypes.begin() != LoopType::Irreducible &&
         "Loop type should not be Irreducible for a block");
  return *loopTypes.begin();
}

const llvm::DenseSet<Block *> &LoopTree::getLoopChildren(Block *block) const {
  if (auto it = loopTree.find(block); it != loopTree.end())
    return it->second;
  // If the block is not a loop header, return an empty set
  static llvm::DenseSet<Block *> emptySet;
  return emptySet;
}

Block *LoopTree::getLoopHeader(Block *block) const {
  if (auto it = loopHeader.find(block); it != loopHeader.end())
    return it->second;
  return nullptr; // entry point of the function
}

struct LoopTreeBuilder {
  using LoopType = LoopAnalysis::LoopType;

  LoopTreeBuilder(Module *module, Function *function,
                  const DominatorTree *domtree)
      : module(module), function(function), dominatorTree(domtree) {}

  void dfsImpl(Block *block);

  void dfs();

  bool isAncestor(size_t ancestor, size_t node) const;

  LoopTree build();

  void analyzeLoops();

  void fixLoops();

  Block *createDummyBlock(Block *original);

  size_t find(size_t node) {
    if (auto it = parent.find(node); it != parent.end())
      return node == it->second ? node : it->second = find(it->second);
    return parent[node] = node;
  }

  void union_(size_t a, size_t b) {
    a = find(a);
    b = find(b);
    if (a == b)
      return;

    auto rankA = getRank(a);
    auto rankB = getRank(b);

    if (rankA < rankB)
      parent[a] = b;
    else {
      parent[b] = a;
      if (rankA == rankB)
        rank[a]++;
    }
  }

  size_t getRank(size_t v) {
    if (auto it = rank.find(v); it != rank.end())
      return it->second;
    return rank[v] = 0;
  }

  Module *module;
  Function *function;
  const DominatorTree *dominatorTree;
  llvm::DenseMap<size_t, size_t> lastSubTreeNode;
  llvm::DenseMap<Block *, size_t> blockOrderMap;
  llvm::DenseMap<size_t, Block *> blockOrderRev;

  llvm::DenseMap<Block *, llvm::DenseSet<Block *>> successors;
  llvm::DenseMap<Block *, llvm::DenseSet<Block *>> predecessors;

  llvm::DenseMap<size_t, llvm::DenseSet<size_t>> backPreds;
  llvm::DenseMap<size_t, llvm::DenseSet<size_t>> nonBackPreds;
  llvm::DenseMap<size_t, size_t> headerMap;
  llvm::DenseMap<size_t, LoopType> loopTypeMap;

  llvm::DenseMap<size_t, size_t> parent;
  llvm::DenseMap<size_t, size_t> rank;

  llvm::SmallVector<std::unique_ptr<Block>> dummyBlocks;
  llvm::DenseMap<Block *, Block *> dummyToOriginal;
  llvm::DenseMap<Block *, Block *> originalToDummy;
  size_t order = 0;
};

void LoopTreeBuilder::dfsImpl(Block *block) {
  blockOrderMap[block] = order;
  blockOrderRev[order] = block;
  order++;

  auto succs = module->getSuccessors(block);
  for (Block *succ : succs) {
    if (!blockOrderMap.contains(succ))
      dfsImpl(succ);
  }
  lastSubTreeNode[blockOrderMap[block]] = order - 1;
}

void LoopTreeBuilder::dfs() {
  auto entryBlock = function->getEntryBlock();
  order = 0;
  lastSubTreeNode.clear();
  blockOrderMap.clear();
  blockOrderRev.clear();
  dfsImpl(entryBlock);
}

bool LoopTreeBuilder::isAncestor(size_t ancestor, size_t node) const {
  return (ancestor <= node && node <= lastSubTreeNode.at(ancestor));
}

LoopTree LoopTreeBuilder::build() {
  // copy cfg from module
  // The graph can be transformed while analyzing loops
  for (Block *block : *function) {
    successors[block] = module->getSuccessors(block);
    predecessors[block] = module->getPredecessors(block);
  }

  // fixLoops
  fixLoops();

  // analyze loops
  analyzeLoops();

  // build impl from the results
  llvm::DenseMap<Block *, llvm::DenseSet<Block *>> loopTree;
  llvm::DenseMap<Block *, Block *> loopHeader;
  llvm::DenseMap<Block *, llvm::DenseSet<LoopType>> loopType;

  for (auto [block, order] : blockOrderMap) {
    if (auto it = dummyToOriginal.find(block); it != dummyToOriginal.end())
      block = it->second;

    auto type = loopTypeMap[order];
    auto header = headerMap[order];
    auto headerBlock = blockOrderRev[header];

    if (auto it = dummyToOriginal.find(headerBlock);
        it != dummyToOriginal.end())
      headerBlock = it->second;

    loopTree[headerBlock].insert(block);
    loopHeader[block] = headerBlock;
    loopType[block].insert(type);
  }

  return LoopTree(std::move(loopTree), std::move(loopHeader),
                  std::move(loopType));
}

Block *LoopTreeBuilder::createDummyBlock(Block *original) {
  auto dummy = new Block(-1);
  dummyBlocks.emplace_back(dummy);

  dummyToOriginal[dummy] = original;
  originalToDummy[original] = dummy;
  return dummy;
}

void LoopTreeBuilder::fixLoops() {
  dfs();

  for (auto curr = 0u; curr < order; ++curr) {
    llvm::DenseSet<size_t> redBackIn;
    llvm::DenseSet<size_t> otherIn;

    auto preds = module->getPredecessors(blockOrderRev[curr]);
    for (auto predBlock : preds) {
      auto pred = blockOrderMap[predBlock];
      (dominatorTree->isDominator(blockOrderRev[curr], predBlock) ? redBackIn
                                                                  : otherIn)
          .insert(pred);
    }
    if (!redBackIn.empty() && otherIn.size() > 1) {
      // create dummy block
      auto *dummy = createDummyBlock(blockOrderRev[curr]);
      // create edge (dummy, curr)

      successors[dummy].insert(blockOrderRev[curr]);
      predecessors[blockOrderRev[curr]].insert(dummy);

      for (auto newPred : otherIn) {
        // create edge (newPred, dummy)
        successors[blockOrderRev[newPred]].insert(dummy);
        predecessors[dummy].insert(blockOrderRev[newPred]);

        // remove edge (newPred, curr)
        successors[blockOrderRev[newPred]].erase(blockOrderRev[curr]);
        predecessors[blockOrderRev[curr]].erase(blockOrderRev[newPred]);
      }
    }
  }
}

void LoopTreeBuilder::analyzeLoops() {
  dfs();

  for (auto curr = 0u; curr < order; ++curr) {
    loopTypeMap[curr] = LoopType::NonHeader;
    headerMap[curr] = 0;

    auto block = blockOrderRev[curr];
    auto preds = predecessors[block];
    for (Block *predBlock : preds) {
      auto pred = blockOrderMap[predBlock];
      (isAncestor(curr, pred) ? backPreds[curr] : nonBackPreds[curr])
          .insert(pred);
    }
  }
  headerMap[0] = std::numeric_limits<size_t>::max();

  for (auto curr = order - 1; curr >= 0; --curr) {
    llvm::DenseSet<size_t> workList;
    for (auto backPred : backPreds[curr]) {
      if (backPred != curr)
        workList.insert(find(backPred));
      else
        loopTypeMap[curr] = LoopType::SelfLoop;
    }
    if (!workList.empty())
      loopTypeMap[curr] = LoopType::Reducible;

    auto savedWorkList = workList;
    while (!workList.empty()) {
      auto begin = workList.begin();
      auto backPred = *begin;
      workList.erase(begin);

      for (auto nonBackPred : nonBackPreds[backPred]) {
        nonBackPred = find(nonBackPred);
        if (!isAncestor(curr, nonBackPred)) {
          loopTypeMap[curr] = LoopType::Irreducible;
          nonBackPreds[curr].insert(nonBackPred);
        } else if (!workList.contains(nonBackPred) && curr != nonBackPred) {
          workList.insert(nonBackPred);
          savedWorkList.insert(nonBackPred);
        }
      }
    }

    for (auto node : savedWorkList) {
      headerMap[node] = curr;
      union_(node, curr);
    }
  }
}

class LoopAnalysisImpl {
public:
  using LoopType = LoopAnalysis::LoopType;

  static std::unique_ptr<LoopAnalysisImpl> create(Module *module);

  const LoopTree *getLoopTree(Function *function) const {
    return &loopTrees.at(function);
  }

private:
  LoopAnalysisImpl(llvm::DenseMap<Function *, LoopTree> loopTrees)
      : loopTrees(std::move(loopTrees)) {}

  llvm::DenseMap<Function *, LoopTree> loopTrees;
};

std::unique_ptr<LoopAnalysisImpl> LoopAnalysisImpl::create(Module *module) {
  llvm::DenseMap<Function *, LoopTree> loopTrees;

  DominanceAnalysis *domAnalysis = module->getAnalysis<DominanceAnalysis>();
  if (!domAnalysis) {
    auto analysis = DominanceAnalysis::create(module);
    module->insertAnalysis(std::move(analysis));
    domAnalysis = module->getAnalysis<DominanceAnalysis>();
  }

  for (Function *function : *module->getIR()) {

    const DominatorTree *domTree = domAnalysis->getDominatorTree(function);

    LoopTreeBuilder builder(module, function, domTree);
    LoopTree loopTree = builder.build();

    loopTrees[function] = std::move(loopTree);
  }

  return std::unique_ptr<LoopAnalysisImpl>(
      new LoopAnalysisImpl(std::move(loopTrees)));
}

LoopAnalysis::LoopAnalysis(Module *module,
                           std::unique_ptr<LoopAnalysisImpl> impl)
    : Analysis(module), impl(std::move(impl)) {}
LoopAnalysis::~LoopAnalysis() = default;

std::unique_ptr<LoopAnalysis> LoopAnalysis::create(Module *module) {
  auto impl = LoopAnalysisImpl::create(module);
  return std::unique_ptr<LoopAnalysis>(
      new LoopAnalysis(module, std::move(impl)));
}

const llvm::DenseSet<LoopAnalysis::LoopType> &
LoopAnalysis::getLoopTypes(Block *block) const {
  auto *loopTree = impl->getLoopTree(block->getParentFunction());
  return loopTree->getLoopTypes(block);
}

LoopAnalysis::LoopType LoopAnalysis::getLoopType(Block *block) const {
  auto *loopTree = impl->getLoopTree(block->getParentFunction());
  return loopTree->getLoopType(block);
}

const llvm::DenseSet<Block *> &
LoopAnalysis::getLoopChildren(Block *block) const {
  auto *loopTree = impl->getLoopTree(block->getParentFunction());
  return loopTree->getLoopChildren(block);
}

Block *LoopAnalysis::getLoopHeader(Block *block) const {
  auto *loopTree = impl->getLoopTree(block->getParentFunction());
  return loopTree->getLoopHeader(block);
}

} // namespace kecc::ir
