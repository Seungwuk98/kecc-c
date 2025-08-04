#ifndef KECC_IR_ANALYSES_H
#define KECC_IR_ANALYSES_H

#include "kecc/ir/Analysis.h"
#include "kecc/ir/Module.h"
#include "kecc/ir/Type.h"
#include "kecc/parser/Lexer.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <set>

namespace kecc::ir {

class DominatorTree {
public:
  DominatorTree() = default;
  DominatorTree(DominatorTree &&other) { *this = std::move(other); };
  DominatorTree &operator=(DominatorTree &&other) {
    domTree = std::move(other.domTree);
    idomMap = std::move(other.idomMap);
    dfAdj = std::move(other.dfAdj);
    dfAdjRev = std::move(other.dfAdjRev);
    dfsOrderMap = std::move(other.dfsOrderMap);
    dfsLastSubTreeMap = std::move(other.dfsLastSubTreeMap);
    return *this;
  };

  DominatorTree(llvm::DenseMap<Block *, llvm::DenseSet<Block *>> domTree,
                llvm::DenseMap<Block *, Block *> idomMap,
                llvm::DenseMap<Block *, llvm::DenseSet<Block *>> dfAdj,
                llvm::DenseMap<Block *, llvm::DenseSet<Block *>> dfAdjRev,
                llvm::DenseMap<Block *, size_t> dfsOrderMap,
                llvm::DenseMap<Block *, size_t> dfsLastSubTreeMap)
      : domTree(std::move(domTree)), idomMap(std::move(idomMap)),
        dfAdj(std::move(dfAdj)), dfAdjRev(std::move(dfAdjRev)),
        dfsOrderMap(std::move(dfsOrderMap)),
        dfsLastSubTreeMap(std::move(dfsLastSubTreeMap)) {}

  Block *getIdom(Block *block) const;

  const llvm::DenseSet<Block *> &getChildren(Block *block) const;

  const llvm::DenseSet<Block *> &getDF(Block *block) const;

  const llvm::DenseSet<Block *> &getDFRev(Block *block) const;

  bool isDominator(Block *dom, Block *block) const;

private:
  llvm::DenseMap<Block *, llvm::DenseSet<Block *>> domTree;
  llvm::DenseMap<Block *, Block *> idomMap;
  llvm::DenseMap<Block *, llvm::DenseSet<Block *>> dfAdj;
  llvm::DenseMap<Block *, llvm::DenseSet<Block *>> dfAdjRev;

  llvm::DenseMap<Block *, size_t> dfsOrderMap;
  llvm::DenseMap<Block *, size_t> dfsLastSubTreeMap;
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

class LiveRangeAnalysisImpl;
class LiveRangeAnalysis : public Analysis {
public:
  ~LiveRangeAnalysis();

  static std::unique_ptr<LiveRangeAnalysis> create(Module *module);

  size_t getLiveRange(Function *func, Value value) const;
  Type getLiveRangeType(Function *func, size_t liveRange) const;

  void dump(llvm::raw_ostream &os) const;

private:
  LiveRangeAnalysis(Module *module,
                    std::unique_ptr<LiveRangeAnalysisImpl> impl);

  std::unique_ptr<LiveRangeAnalysisImpl> impl;
};

class LivenessAnalysis : public Analysis {
public:
  static std::unique_ptr<LivenessAnalysis> create(Module *module);

  void dump(llvm::raw_ostream &os) const;

  const std::set<size_t> &getLiveVars(Block *block) const;

private:
  LivenessAnalysis(Module *module,
                   llvm::DenseMap<Block *, std::set<size_t>> liveOut)
      : Analysis(module), liveOut(std::move(liveOut)) {}

  const llvm::DenseMap<Block *, std::set<size_t>> liveOut;
};

class LoopAnalysisImpl;
class LoopAnalysis : public Analysis {
public:
  ~LoopAnalysis();

  enum LoopType {
    NonHeader,
    Reducible,
    Irreducible,
    SelfLoop,
  };

  static std::unique_ptr<LoopAnalysis> create(Module *module);

  LoopType getLoopType(Block *header) const;
  const llvm::DenseSet<Block *> &getLoopChildren(Block *block) const;
  Block *getLoopHeader(Block *block) const;

  bool hasIrreducibleLoops() const;

  void dump(llvm::raw_ostream &os) const;

private:
  LoopAnalysis(Module *module, std::unique_ptr<LoopAnalysisImpl> impl);
  std::unique_ptr<LoopAnalysisImpl> impl;
};

} // namespace kecc::ir

namespace llvm {

template <>
struct DenseMapInfo<kecc::ir::LoopAnalysis::LoopType>
    : public llvm::DenseMapInfo<unsigned> {};

} // namespace llvm

DECLARE_KECC_TYPE_ID(kecc::ir::DominanceAnalysis)
DECLARE_KECC_TYPE_ID(kecc::ir::VisitOrderAnalysis)
DECLARE_KECC_TYPE_ID(kecc::ir::LiveRangeAnalysis)
DECLARE_KECC_TYPE_ID(kecc::ir::LivenessAnalysis)
DECLARE_KECC_TYPE_ID(kecc::ir::LoopAnalysis)

#endif // KECC_IR_ANALYSES_H
