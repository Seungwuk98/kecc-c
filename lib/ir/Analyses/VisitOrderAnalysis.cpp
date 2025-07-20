#include "kecc/ir/IRAnalyses.h"

DEFINE_KECC_TYPE_ID(kecc::ir::VisitOrderAnalysis);

namespace kecc::ir {

namespace {
class DFS {
public:
  enum Order { PreOrder, PostOrder };

  template <Order OrderType>
  static std::unique_ptr<DFS> create(Module *module, Function *func) {
    auto dfs = std::unique_ptr<DFS>(new DFS(module));
    dfs->dfs<OrderType>(func->getEntryBlock());
    return dfs;
  }

  llvm::ArrayRef<Block *> getOrder() const { return order; }

private:
  DFS(Module *module) : module(module) {}

  template <Order OrderType> void dfs(Block *block) {
    if (visited.contains(block))
      return;
    visited.insert(block);

    if constexpr (OrderType == PreOrder) {
      order.emplace_back(block);
    }

    auto successors = module->getSuccessors(block);
    for (Block *succ : successors) {
      dfs<OrderType>(succ);
    }

    if constexpr (OrderType == PostOrder) {
      order.emplace_back(block);
    }
  }

  Module *module;
  std::vector<Block *> order;
  llvm::DenseSet<Block *> visited;
};
} // namespace

std::unique_ptr<VisitOrderAnalysis> VisitOrderAnalysis::create(Module *module) {
  llvm::DenseMap<Function *,
                 std::pair<std::vector<Block *>, std::vector<Block *>>>
      postOrderMap;
  llvm::DenseMap<Function *,
                 std::pair<std::vector<Block *>, std::vector<Block *>>>
      preOrderMap;
  for (Function *func : *module->getIR()) {
    auto preDFS = DFS::create<DFS::PreOrder>(module, func);
    auto preOrder = preDFS->getOrder();
    std::vector<Block *> reversePreOrder(preOrder.rbegin(), preOrder.rend());
    auto postDFS = DFS::create<DFS::PostOrder>(module, func);
    auto postOrder = postDFS->getOrder();
    std::vector<Block *> reversePostOrder(postOrder.rbegin(), postOrder.rend());
    preOrderMap[func] = {preOrder, reversePreOrder};
    postOrderMap[func] = {postOrder, reversePostOrder};
  }

  return std::unique_ptr<VisitOrderAnalysis>(new VisitOrderAnalysis(
      module, std::move(postOrderMap), std::move(preOrderMap)));
}

} // namespace kecc::ir
