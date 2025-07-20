#ifndef KECC_IR_ANALYSES_H
#define KECC_IR_ANALYSES_H

#include "kecc/ir/Analysis.h"
#include "kecc/ir/Module.h"
#include "kecc/ir/Type.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

namespace kecc::ir {

class DominatorTree {
public:
  DominatorTree() = default;
  DominatorTree(DominatorTree &&other)
      : domTree(std::move(other.domTree)), idomMap(std::move(other.idomMap)),
        dfAdj(std::move(other.dfAdj)), dfAdjRev(std::move(other.dfAdjRev)) {};
  DominatorTree &operator=(DominatorTree &&other) {
    domTree = std::move(other.domTree);
    idomMap = std::move(other.idomMap);
    dfAdj = std::move(other.dfAdj);
    dfAdjRev = std::move(other.dfAdjRev);
    return *this;
  };

  DominatorTree(llvm::DenseMap<Block *, llvm::DenseSet<Block *>> domTree,
                llvm::DenseMap<Block *, Block *> idomMap,
                llvm::DenseMap<Block *, llvm::DenseSet<Block *>> dfAdj,
                llvm::DenseMap<Block *, llvm::DenseSet<Block *>> dfAdjRev)
      : domTree(std::move(domTree)), idomMap(std::move(idomMap)),
        dfAdj(std::move(dfAdj)), dfAdjRev(std::move(dfAdjRev)) {}

  Block *getIdom(Block *block) const;

  const llvm::DenseSet<Block *> &getChildren(Block *block) const;

  const llvm::DenseSet<Block *> &getDF(Block *block) const;

  const llvm::DenseSet<Block *> &getDFRev(Block *block) const;

private:
  llvm::DenseMap<Block *, llvm::DenseSet<Block *>> domTree;
  llvm::DenseMap<Block *, Block *> idomMap;
  llvm::DenseMap<Block *, llvm::DenseSet<Block *>> dfAdj;
  llvm::DenseMap<Block *, llvm::DenseSet<Block *>> dfAdjRev;
};

class DominanceAnalysis : public Analysis {
public:
  static std::unique_ptr<DominanceAnalysis> create(Module *module);

  const DominatorTree *getDominatorTree(Function *function) const {
    auto it = dominatorTrees.find(function);
    if (it != dominatorTrees.end()) {
      return &it->second;
    }
    return nullptr;
  }

  void dump(llvm::raw_ostream &os) const;

private:
  DominanceAnalysis(Module *module,
                    llvm::DenseMap<Function *, DominatorTree> domTree)
      : Analysis(module), dominatorTrees(std::move(domTree)) {}

  llvm::DenseMap<Function *, DominatorTree> dominatorTrees;
};

class VisitOrderAnalysis : public Analysis {
public:
  static std::unique_ptr<VisitOrderAnalysis> create(Module *module);

  llvm::ArrayRef<Block *> getPostOrder(Function *function) const {
    auto it = postOrderMap.find(function);
    if (it != postOrderMap.end()) {
      return it->second.first;
    }
    return {};
  }

  llvm::ArrayRef<Block *> getReversePostOrder(Function *function) const {
    auto it = postOrderMap.find(function);
    if (it != postOrderMap.end()) {
      return it->second.second;
    }
    return {};
  }

  llvm::ArrayRef<Block *> getPreOrder(Function *function) const {
    auto it = preOrderMap.find(function);
    if (it != preOrderMap.end()) {
      return it->second.first;
    }
    return {};
  }

  llvm::ArrayRef<Block *> getReversePreOrder(Function *function) const {
    auto it = preOrderMap.find(function);
    if (it != preOrderMap.end()) {
      return it->second.second;
    }
    return {};
  }

private:
  VisitOrderAnalysis(
      Module *module,
      llvm::DenseMap<Function *,
                     std::pair<std::vector<Block *>, std::vector<Block *>>>
          revPostOrderMap,
      llvm::DenseMap<Function *,
                     std::pair<std::vector<Block *>, std::vector<Block *>>>
          revPreOrderMap)
      : Analysis(module), postOrderMap(std::move(revPostOrderMap)),
        preOrderMap(std::move(revPreOrderMap)) {}
  llvm::DenseMap<Function *,
                 std::pair<std::vector<Block *>, std::vector<Block *>>>
      postOrderMap;
  llvm::DenseMap<Function *,
                 std::pair<std::vector<Block *>, std::vector<Block *>>>
      preOrderMap;
};

} // namespace kecc::ir

DECLARE_KECC_TYPE_ID(kecc::ir::DominanceAnalysis)
DECLARE_KECC_TYPE_ID(kecc::ir::VisitOrderAnalysis)

#endif // KECC_IR_ANALYSES_H
