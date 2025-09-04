#ifndef KECC_TRANSLATE_MOVE_SCHEDULE_H
#define KECC_TRANSLATE_MOVE_SCHEDULE_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include <queue>
#include <set>

namespace kecc {

enum class Movement {
  Move,
  Swap,
};

template <typename T> class MoveManagement {
public:
  MoveManagement(llvm::ArrayRef<T> dst, llvm::ArrayRef<T> src, T temp);

  llvm::ArrayRef<std::tuple<Movement, T, T>> getMoveSchedule() const {
    return moveSchedule;
  }

  void dump(llvm::raw_ostream &os) const {
    for (const auto &[movement, dst, src] : moveSchedule) {
      if (movement == Movement::Move)
        os << "Move " << dst << " <- " << src << "\n";
      else
        os << "Swap " << dst << " <-> " << src << "\n";
    }
  }

private:
  void swap(int a, int b);
  void copy(int dst, int src);
  void move(int dst, int src);
  void init(llvm::ArrayRef<T> dst, llvm::ArrayRef<T> src, T temp);

  llvm::DenseMap<T, int> nodeToIndex;
  llvm::DenseMap<int, T> indexToNode;
  std::set<int> unresolvedNode;
  llvm::SmallVector<std::set<int>> moveGraph;
  llvm::SmallVector<int> moveGraphRev;
  llvm::SmallVector<std::tuple<Movement, T, T>> moveSchedule;
};

template <typename T>
MoveManagement<T>::MoveManagement(llvm::ArrayRef<T> dst, llvm::ArrayRef<T> src,
                                  T temp) {
  init(dst, src, temp);
}

template <typename T> void MoveManagement<T>::swap(int a, int b) {
  assert(a != b && "a and b must be different");
  auto nodeA = indexToNode.at(a);
  auto nodeB = indexToNode.at(b);
  if constexpr (requires {
                  { nodeA < nodeB } -> std::convertible_to<bool>;
                }) {
    if (nodeA > nodeB)
      std::swap(nodeA, nodeB);
  }
  moveSchedule.emplace_back(Movement::Swap, nodeA, nodeB);

  // update graph
  moveGraphRev[a] = moveGraphRev[b] = -1;
  unresolvedNode.erase(a);
  unresolvedNode.erase(b);

  auto moveA = moveGraph[a];
  auto moveB = moveGraph[b];
  moveA.erase(b);
  moveB.erase(a);
  moveGraph[a] = moveB;
  moveGraph[b] = moveA;
  for (int dst : moveGraph[a])
    moveGraphRev[dst] = a;
  for (int dst : moveGraph[b])
    moveGraphRev[dst] = b;
}

template <typename T> void MoveManagement<T>::copy(int dst, int src) {
  assert(dst != src && "dst and src must be different");
  moveSchedule.emplace_back(Movement::Move, indexToNode.at(dst),
                            indexToNode.at(src));

  // update graph
  moveGraphRev[dst] = -1;
  unresolvedNode.erase(dst);
  moveGraph[src].erase(dst);
}

template <typename T> void MoveManagement<T>::move(int dst, int src) {
  assert(dst != src && "dst and src must be different");
  moveSchedule.emplace_back(Movement::Move, indexToNode.at(dst),
                            indexToNode.at(src));

  // update graph
  for (int originalDsts : moveGraph[src]) {
    moveGraphRev[originalDsts] = dst;
    moveGraph[dst].insert(originalDsts);
  }
  moveGraph[src].clear();
  moveGraphRev[dst] = -1;
}

template <typename T>
void MoveManagement<T>::init(llvm::ArrayRef<T> dst, llvm::ArrayRef<T> src,
                             T temp) {
  assert(dst.size() == src.size() && "dst and src must have the same size");

  int index = 0;
  indexToNode.try_emplace(index, temp);
  nodeToIndex[temp] = index++;

  for (const auto &d : dst) {
    if (nodeToIndex.contains(d))
      continue;

    indexToNode.try_emplace(index, d);
    nodeToIndex[d] = index++;
  }

  for (const auto &s : src) {
    if (nodeToIndex.contains(s))
      continue;

    indexToNode.try_emplace(index, s);
    nodeToIndex[s] = index++;
  }

  moveGraph.resize(index);
  moveGraphRev.resize(index, -1);

  for (size_t i = 0; i < dst.size(); ++i) {
    int d = nodeToIndex[dst[i]];
    int s = nodeToIndex[src[i]];

    if (d == s)
      continue;

    moveGraph[s].insert(d);
    moveGraphRev[d] = s;
    unresolvedNode.insert(d);
  }

  // find swap
  for (size_t i = 0; i < dst.size(); ++i) {
    if (!unresolvedNode.contains(i))
      continue;

    int j = moveGraphRev[i];
    if (!moveGraph[i].contains(j))
      continue;
    assert(j != -1 && "src must exist");
    if (moveGraph[j].contains(i))
      swap(i, j);
  }

  std::queue<int> zeroOutUnresolved;

  for (int node : unresolvedNode) {
    if (moveGraph[node].empty())
      zeroOutUnresolved.push(node);
  }

  while (!unresolvedNode.empty()) {
    if (zeroOutUnresolved.empty()) {
      // cycle exists
      auto firstIter = unresolvedNode.begin();
      int startNode = *firstIter;
      move(0, startNode);
      zeroOutUnresolved.push(startNode);
    }

    while (!zeroOutUnresolved.empty()) {
      int node = zeroOutUnresolved.front();
      zeroOutUnresolved.pop();
      assert(unresolvedNode.contains(node) && "node must be in unresolvedNode");

      int src = moveGraphRev[node];
      assert(src != -1 && "src must exist");
      copy(node, src);

      moveGraphRev[node] = -1;
      unresolvedNode.erase(node);
      if (moveGraph[src].empty() && unresolvedNode.contains(src))
        zeroOutUnresolved.push(src);
    }
  }
}

} // namespace kecc

#endif // KECC_TRANSLATE_MOVE_SCHEDULE_H
