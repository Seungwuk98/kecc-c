#include "kecc/ir/IR.h"
#include "kecc/ir/IRAnalyses.h"

DEFINE_KECC_TYPE_ID(kecc::ir::DominanceAnalysis);

namespace kecc::ir {

Block *DominatorTree::getIdom(Block *block) const {
  if (block->getParentFunction()->getEntryBlock() == block) {
    return nullptr; // Entry block has no immediate dominator
  }
  return idomMap.at(block);
}

const llvm::DenseSet<Block *> &DominatorTree::getChildren(Block *block) const {
  return domTree.at(block);
}

const llvm::DenseSet<Block *> &DominatorTree::getDF(Block *block) const {
  return dfAdj.at(block);
}

const llvm::DenseSet<Block *> &DominatorTree::getDFRev(Block *block) const {
  return dfAdjRev.at(block);
}

bool DominatorTree::isDominator(Block *dom, Block *block) const {
  auto domOrder = dfsOrderMap.at(dom);
  auto blockOrder = dfsOrderMap.at(block);

  return domOrder <= blockOrder && blockOrder <= dfsLastSubTreeMap.at(dom);
}

struct DominaceAnalysisBuilder {
  Module *module;
  Function *function;
  llvm::DenseMap<Block *, llvm::DenseSet<Block *>> tree;
  llvm::DenseMap<Block *, Block *> idomBlockMap;
  llvm::DenseMap<Block *, llvm::DenseSet<Block *>> dfAdj;
  llvm::DenseMap<Block *, llvm::DenseSet<Block *>> dfAdjRev;

  llvm::DenseMap<int64_t, llvm::DenseSet<int64_t>> bucket;
  llvm::DenseMap<int64_t, int64_t> idomMap;
  llvm::DenseMap<int64_t, int64_t> sdomMap;
  llvm::DenseMap<int64_t, llvm::DenseSet<int64_t>> revAdj;
  llvm::DenseMap<int64_t, int64_t> parent;
  llvm::DenseMap<int64_t, int64_t> dsuParent;
  llvm::DenseMap<int64_t, int64_t> label;

  llvm::DenseMap<Block *, int64_t> newIdx;
  llvm::DenseMap<int64_t, Block *> idxToBlock;

  llvm::DenseMap<Block *, size_t> domTreeOrder;
  llvm::DenseMap<Block *, size_t> domTreeLastSubTree;

  int64_t order = 0;

  DominaceAnalysisBuilder(Module *module, Function *function)
      : module(module), function(function) {}

  void init();
  void dfs(Block *block);
  int64_t find(int64_t curr, bool init);
  void union_(int64_t curr, int64_t par);
  void constructSdom();
  void constructIdom();
  void buildDF();

  void dfsDomTree(Block *block);

  DominatorTree build();
};

void DominaceAnalysisBuilder::init() {
  for (Block *block : *function) {
    tree.try_emplace(block, llvm::DenseSet<Block *>());
    dfAdj.try_emplace(block, llvm::DenseSet<Block *>());
    dfAdjRev.try_emplace(block, llvm::DenseSet<Block *>());
  }
  Block *entryBlock = function->getEntryBlock();
  dfs(entryBlock);
}

void DominaceAnalysisBuilder::dfs(Block *block) {
  order++;

  newIdx[block] = dsuParent[order] = sdomMap[order] = label[order] = order;
  idxToBlock[order] = block;

  for (auto *succ : module->getSuccessors(block)) {
    if (!newIdx.contains(succ)) {
      dfs(succ);
      parent[newIdx[succ]] = newIdx[block];
    }
    revAdj[newIdx[succ]].insert(newIdx[block]);
  }
}

int64_t DominaceAnalysisBuilder::find(int64_t curr, bool init) {
  if (curr == dsuParent[curr])
    return init ? curr : -1;

  auto par = find(dsuParent[curr], false);
  if (par == -1)
    return curr;

  if (sdomMap[label[dsuParent[curr]]] < sdomMap[label[curr]])
    label[curr] = label[dsuParent[curr]];

  dsuParent[curr] = par;
  return init ? label[curr] : par;
}

void DominaceAnalysisBuilder::union_(int64_t curr, int64_t par) {
  dsuParent[curr] = par;
}

void DominaceAnalysisBuilder::constructSdom() {

  for (int64_t curr = order; curr > 0; --curr) {
    for (auto pred : revAdj[curr]) {
      auto currSdom = sdomMap[curr];
      auto predFind = find(pred, true);
      auto predSdom = sdomMap[predFind];
      sdomMap[curr] = std::min(currSdom, predSdom);
    }

    if (curr > 1)
      bucket[sdomMap[curr]].insert(curr);

    for (auto cand : bucket[curr]) {
      auto minSdomNode = find(cand, true);
      if (sdomMap[minSdomNode] == curr)
        idomMap[cand] = sdomMap[cand];
      else
        idomMap[cand] = minSdomNode;
    }

    if (curr > 1)
      union_(curr, parent[curr]);
  }
}

void DominaceAnalysisBuilder::constructIdom() {
  for (int64_t curr = 1; curr <= order; ++curr) {
    if (!idomMap.contains(curr))
      continue;
    if (idomMap[curr] != sdomMap[curr])
      idomMap[curr] = idomMap[idomMap[curr]];

    tree[idxToBlock[idomMap[curr]]].insert(idxToBlock[curr]);
    idomBlockMap[idxToBlock[curr]] = idxToBlock[idomMap[curr]];
  }
}

void DominaceAnalysisBuilder::buildDF() {
  for (Block *curr : *function) {
    if (module->getPredecessors(curr).size() < 2)
      continue;

    for (Block *pred : module->getPredecessors(curr)) {
      Block *runner = pred;

      while (runner && runner != idomBlockMap[curr]) {
        dfAdj[runner].insert(curr);
        runner = idomBlockMap[runner];
      }
    }
  }

  for (auto [block, dfSet] : dfAdj) {
    for (Block *dfBlock : dfSet) {
      dfAdjRev[dfBlock].insert(block);
    }
  }
}

void DominaceAnalysisBuilder::dfsDomTree(Block *block) {
  domTreeOrder[block] = order++;
  for (Block *child : tree[block]) {
    if (!domTreeOrder.contains(child))
      dfsDomTree(child);
  }
  domTreeLastSubTree[block] = order - 1;
}

DominatorTree DominaceAnalysisBuilder::build() {
  init();
  constructSdom();
  constructIdom();
  buildDF();

  order = 0;
  dfsDomTree(function->getEntryBlock());

  return DominatorTree(std::move(tree), std::move(idomBlockMap),
                       std::move(dfAdj), std::move(dfAdjRev),
                       std::move(domTreeOrder), std::move(domTreeLastSubTree));
}

std::unique_ptr<DominanceAnalysis> DominanceAnalysis::create(Module *module) {
  auto ir = module->getIR();
  llvm::DenseMap<Function *, DominatorTree> dominatorTrees;
  for (auto *function : *ir) {
    DominaceAnalysisBuilder builder(module, function);
    auto dominatorTree = builder.build();
    dominatorTrees.try_emplace(function, std::move(dominatorTree));
  }

  return std::unique_ptr<DominanceAnalysis>(
      new DominanceAnalysis(module, std::move(dominatorTrees)));
}

void DominanceAnalysis::dump(llvm::raw_ostream &os) const {
  os << "Dominator Analysis dump\n";

  llvm::SmallVector<Function *> functions;
  functions.reserve(dominatorTrees.size());
  for (const auto &[function, _] : dominatorTrees)
    functions.emplace_back(function);

  llvm::sort(functions, [](Function *lhs, Function *rhs) {
    return lhs->getName() < rhs->getName();
  });

  for (auto *function : functions) {
    auto domTree = getDominatorTree(function);
    assert(domTree && "Dominator tree should not be null");

    os << "Domtree (@" << function->getName() << "):";

    llvm::SmallVector<Block *> blocks;
    for (Block *block : *function)
      blocks.emplace_back(block);
    llvm::sort(blocks, [](Block *lhs, Block *rhs) {
      return lhs->getId() < rhs->getId();
    });

    for (Block *block : blocks) {
      os << "\nb" << block->getId() << ":";
      if (!domTree->getChildren(block).empty())
        os << ' ';

      llvm::SmallVector<Block *> children(domTree->getChildren(block).begin(),
                                          domTree->getChildren(block).end());
      llvm::sort(children, [](Block *lhs, Block *rhs) {
        return lhs->getId() < rhs->getId();
      });

      for (auto I = children.begin(), E = children.end(); I != E; ++I) {
        if (I != children.begin()) {
          os << ", ";
        }
        os << 'b' << (*I)->getId();
      }
    }

    os << "\n\nImmediate dominators:\n";
    for (Block *block : blocks) {
      auto idom = domTree->getIdom(block);
      if (idom) {
        os << "b" << block->getId() << " -> b" << idom->getId() << "\n";
      } else {
        os << "b" << block->getId() << " -> <none>\n";
      }
    }

    os << "\n\nDominance frontier:\n";
    for (Block *block : blocks) {
      os << "b" << block->getId() << ":";
      const auto &df = domTree->getDF(block);
      if (!df.empty()) {
        os << ' ';
        llvm::SmallVector<Block *> dfBlocks(df.begin(), df.end());
        llvm::sort(dfBlocks, [](Block *lhs, Block *rhs) {
          return lhs->getId() < rhs->getId();
        });

        for (auto I = dfBlocks.begin(), E = dfBlocks.end(); I != E; ++I) {
          if (I != dfBlocks.begin()) {
            os << ", ";
          }
          os << 'b' << (*I)->getId();
        }
      } else {
        os << " <empty>";
      }
      os << '\n';
    }
  }
}

} // namespace kecc::ir
