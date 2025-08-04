#include "kecc/ir/Analysis.h"
#include "kecc/ir/IRAnalyses.h"
#include "kecc/ir/Type.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

DEFINE_KECC_TYPE_ID(kecc::ir::LoopAnalysis)

namespace kecc::ir {

class LoopTree {
public:
  using LoopType = LoopAnalysis::LoopType;
  LoopTree(Block *entry,
           llvm::DenseMap<Block *, llvm::DenseSet<Block *>> loopTree,
           llvm::DenseMap<Block *, Block *> loopHeader,
           llvm::DenseMap<Block *, LoopType> loopType,
           llvm::SmallVector<std::unique_ptr<Block>> dummyBlocks,
           llvm::DenseMap<Block *, Block *> dummyToOriginal,
           llvm::DenseMap<Block *, Block *> originalToDummy)
      : loopTree(std::move(loopTree)), loopHeader(std::move(loopHeader)),
        loopType(std::move(loopType)), dummyBlocks(std::move(dummyBlocks)),
        dummyToOriginal(std::move(dummyToOriginal)),
        originalToDummy(std::move(originalToDummy)) {
    loopDepth[entry] = 0;
    calcLoopDepth(entry);
  }

  LoopTree() = default;
  LoopTree(const LoopTree &other) = default;
  LoopTree &operator=(const LoopTree &other) = default;
  LoopTree(LoopTree &&other) = default;
  LoopTree &operator=(LoopTree &&other) = default;

  // Returns the loop type of the block.
  // Usually the program is reducible, so it returns NonHeader, Reducible, or
  // SelfLoop.
  LoopType getLoopType(Block *block) const;

  const llvm::DenseSet<Block *> &getLoopChildren(Block *block) const;
  Block *getLoopHeader(Block *block) const;

  void dump(llvm::raw_ostream &os) const;

  Block *getDummyBlock(Block *original) const;
  Block *getOriginalBlock(Block *dummy) const;

  size_t getLoopDepth(Block *block) const { return loopDepth.at(block); }

private:
  void calcLoopDepth(Block *block);

  llvm::DenseMap<Block *, llvm::DenseSet<Block *>> loopTree;
  llvm::DenseMap<Block *, Block *> loopHeader;
  llvm::DenseMap<Block *, LoopType> loopType;

  llvm::DenseMap<Block *, size_t> loopDepth;

  llvm::SmallVector<std::unique_ptr<Block>> dummyBlocks;
  llvm::DenseMap<Block *, Block *> dummyToOriginal;
  llvm::DenseMap<Block *, Block *> originalToDummy;
};

LoopTree::LoopType LoopTree::getLoopType(Block *block) const {
  return loopType.at(block);
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

Block *LoopTree::getDummyBlock(Block *original) const {
  if (auto it = originalToDummy.find(original); it != originalToDummy.end())
    return it->second;
  return nullptr; // no dummy block for the original block
}

Block *LoopTree::getOriginalBlock(Block *dummy) const {
  if (auto it = dummyToOriginal.find(dummy); it != dummyToOriginal.end())
    return it->second;
  return nullptr; // no original block for the dummy block
}

void LoopTree::dump(llvm::raw_ostream &os) const {
  os << "Loop Tree Dump:\n";

  llvm::SmallVector<Block *> blocks;
  blocks.reserve(loopType.size());
  for (const auto &[block, _] : loopType)
    blocks.emplace_back(block);

  auto blockComparator = [&](Block *lhs, Block *rhs) {
    auto lhsId = dummyToOriginal.contains(lhs)
                     ? dummyToOriginal.at(lhs)->getId()
                     : lhs->getId();
    auto rhsId = dummyToOriginal.contains(rhs)
                     ? dummyToOriginal.at(rhs)->getId()
                     : rhs->getId();
    if (lhsId == rhsId) {
      return !dummyToOriginal.contains(lhs);
    }
    return lhsId < rhsId;
  };

  auto blockPrinter = [&](Block *block) -> llvm::raw_ostream & {
    if (auto it = dummyToOriginal.find(block); it != dummyToOriginal.end())
      os << "dummy of block b" << it->second->getId();
    else
      os << "block b" << block->getId();

    return os;
  };

  llvm::sort(blocks, blockComparator);

  bool first = true;
  for (Block *block : blocks) {
    if (first)
      first = false;
    else
      os << "\n";

    blockPrinter(block) << ":\n";
    os << "  type: ";
    switch (loopType.at(block)) {
    case LoopType::NonHeader:
      os << "NonHeader\n";
      break;
    case LoopType::Reducible:
      os << "Reducible\n";
      break;
    case LoopType::Irreducible:
      os << "Irreducible\n";
      break;
    case LoopType::SelfLoop:
      os << "SelfLoop\n";
      break;
    }

    os << "  depth: " << getLoopDepth(block) << '\n';

    if (auto it = loopHeader.find(block); it != loopHeader.end()) {
      os << "  header: ";
      blockPrinter(it->second) << '\n';
    }

    if (auto it = loopTree.find(block); it != loopTree.end()) {
      os << "  children: ";
      const auto &children = it->second;
      llvm::SmallVector<Block *> sortedChildren(children.begin(),
                                                children.end());
      llvm::sort(sortedChildren, blockComparator);

      for (auto I = sortedChildren.begin(), E = sortedChildren.end(); I != E;
           ++I) {
        if (I != sortedChildren.begin())
          os << ", ";
        blockPrinter(*I);
      }
      os << '\n';
    }
  }
}

void LoopTree::calcLoopDepth(Block *block) {
  auto currLoopType = loopType.at(block);
  bool isLoop = (currLoopType == LoopType::Reducible ||
                 currLoopType == LoopType::Irreducible ||
                 currLoopType == LoopType::SelfLoop);
  auto currDepth = loopDepth.at(block);
  if (auto it = loopTree.find(block); it != loopTree.end()) {
    for (Block *child : it->second) {
      auto nextDepth = currDepth + isLoop;
      auto [_, inserted] = loopDepth.try_emplace(child, nextDepth);
      assert(inserted &&
             "Loop depth for child block should not be already set");
      calcLoopDepth(child);
    }
  }
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

  // merge a's set into b's set
  void union_(size_t a, size_t b) {
    a = find(a);
    b = find(b);
    parent[a] = b;
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
  llvm::DenseMap<size_t, size_t> loopDepth;

  llvm::SmallVector<std::unique_ptr<Block>> dummyBlocks;
  llvm::DenseMap<Block *, Block *> dummyToOriginal;
  llvm::DenseMap<Block *, Block *> originalToDummy;

  size_t order = 0;
};

void LoopTreeBuilder::dfsImpl(Block *block) {
  blockOrderMap[block] = order;
  blockOrderRev[order] = block;
  order++;

  // This sorting is not strictly necessary, but it helps to determine the shape
  // of DFS spanning tree.
  // And it is helpful for debugging and analyze loop tree structure.
  llvm::SmallVector<Block *> succs(successors[block].begin(),
                                   successors[block].end());
  llvm::stable_sort(succs, [&](Block *lhs, Block *rhs) {
    auto lhsId = dummyToOriginal.contains(lhs)
                     ? dummyToOriginal.at(lhs)->getId()
                     : lhs->getId();
    auto rhsId = dummyToOriginal.contains(rhs)
                     ? dummyToOriginal.at(rhs)->getId()
                     : rhs->getId();
    return lhsId < rhsId;
  });

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
  llvm::DenseMap<Block *, LoopType> loopType;

  for (auto [block, order] : blockOrderMap) {
    auto type = loopTypeMap[order];
    auto header = headerMap[order];

    if (header != std::numeric_limits<size_t>::max()) {
      auto headerBlock = blockOrderRev[header];
      loopTree[headerBlock].insert(block);
      loopHeader[block] = headerBlock;
    }

    loopType[block] = type;
  }

  return LoopTree(function->getEntryBlock(), std::move(loopTree),
                  std::move(loopHeader), std::move(loopType),
                  std::move(dummyBlocks), std::move(dummyToOriginal),
                  std::move(originalToDummy));
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

    auto preds = predecessors[blockOrderRev[curr]];
    for (auto predBlock : preds) {
      auto pred = blockOrderMap[predBlock];
      (dominatorTree->isDominator(blockOrderRev[curr], predBlock) ? redBackIn
                                                                  : otherIn)
          .insert(pred);
    }
    if (!redBackIn.empty() && otherIn.size() > 1) {
      // create dummy block
      auto *dummy = createDummyBlock(blockOrderRev[curr]);
      auto *currBlock = blockOrderRev[curr];

      // create edge (dummy, curr)
      successors[dummy].insert(currBlock);
      predecessors[currBlock].insert(dummy);

      for (auto pred : otherIn) {
        Block *predBlock = blockOrderRev[pred];
        // create edge (newPred, dummy)
        Block *newPredBlock = blockOrderRev[pred];
        successors[predBlock].insert(dummy);
        predecessors[dummy].insert(predBlock);

        // remove edge (newPred, curr)
        successors[predBlock].erase(currBlock);
        predecessors[currBlock].erase(predBlock);
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

  for (int curr = order - 1; curr >= 0; --curr) {
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
        } else if (!savedWorkList.contains(nonBackPred) &&
                   curr != nonBackPred) {
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
    if (!function->hasDefinition())
      continue;

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

size_t LoopAnalysis::getLoopDepth(Block *block) const {
  auto *loopTree = impl->getLoopTree(block->getParentFunction());
  return loopTree->getLoopDepth(block);
}

void LoopAnalysis::dump(llvm::raw_ostream &os) const {
  os << "Loop Analysis Dump:\n";

  for (Function *function : *getModule()->getIR()) {
    if (!function->hasDefinition())
      continue;

    auto *loopTree = impl->getLoopTree(function);
    os << "function @" << function->getName() << ":\n";
    loopTree->dump(os);

    os << '\n';
  }
}

} // namespace kecc::ir
