#ifndef KECC_TRANSLATE_MOVE_SCHEDULE_H
#define KECC_TRANSLATE_MOVE_SCHEDULE_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
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

private:
  void swap(int a, int b);
  void move(int dst, int src);
  void init(llvm::ArrayRef<T> dst, llvm::ArrayRef<T> src, T temp);

  llvm::DenseMap<T, int> nodeToIndex;
  llvm::DenseMap<int, T> indexToNode;
  std::set<int> unresolvedNode;
  llvm::SmallVector<int> moveGraph;
  llvm::SmallVector<int> moveGraphRev;
  llvm::SmallVector<std::tuple<Movement, T, T>> moveSchedule;
};

template <typename T>
MoveManagement<T>::MoveManagement(llvm::ArrayRef<T> dst, llvm::ArrayRef<T> src,
                                  T temp) {
  init(dst, src, temp);
}

template <typename T> void MoveManagement<T>::swap(int a, int b) {
  assert(nodeToIndex.contains(indexToNode.at(a)) &&
         "a must be a valid node index");
  assert(nodeToIndex.contains(indexToNode.at(b)) &&
         "b must be a valid node index");

  moveSchedule.emplace_back(Movement::Swap, indexToNode.at(a),
                            indexToNode.at(b));
  moveGraph[a] = moveGraph[b] = -1;
  moveGraphRev[a] = moveGraphRev[b] = -1;
  unresolvedNode.erase(a);
  unresolvedNode.erase(b);
}

template <typename T> void MoveManagement<T>::move(int dst, int src) {
  assert(nodeToIndex.contains(indexToNode.at(dst)) &&
         "dst must be a valid node index");
  assert(nodeToIndex.contains(indexToNode.at(src)) &&
         "src must be a valid node index");
  assert(moveGraph[dst] == -1 && "dst must not have any move scheduled to it");

  moveSchedule.emplace_back(Movement::Move, indexToNode.at(dst),
                            indexToNode.at(src));
  moveGraph[src] = -1;
  moveGraphRev[dst] = -1;
  unresolvedNode.erase(dst);
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

  moveGraph.resize(index, -1);
  moveGraphRev.resize(index, -1);

  for (auto [d, s] : llvm::zip(dst, src)) {
    if (d == s)
      continue;

    int dIndex = nodeToIndex[d];
    int sIndex = nodeToIndex[s];

    moveGraph[sIndex] = dIndex;
    moveGraphRev[dIndex] = sIndex;
  }

  for (int i = 0; i < index; ++i) {
    if (moveGraphRev[i] != -1)
      unresolvedNode.insert(i);
  }

  for (int i = 0; i < index; ++i) {
    if (!unresolvedNode.contains(i) || moveGraph[i] == -1)
      continue;

    int j = moveGraph[i];
    if (j == -1)
      continue;

    if (unresolvedNode.contains(j) && moveGraph[j] == i)
      swap(i, j);
  }

  llvm::SmallVector<int> stack;
  for (int i = 0; i < index; ++i) {
    if (!unresolvedNode.contains(i))
      continue;

    if (moveGraph[i] == -1 && moveGraphRev[i] != -1) {
      stack.emplace_back(i);
    }
  }

  while (!unresolvedNode.empty()) {
    if (stack.empty()) {
      // break a loop
      auto it = unresolvedNode.begin();
      int i = *it;
      stack.emplace_back(i);

      auto imove = moveGraph[i];
      move(0, i); // move data to temp
      moveGraph[0] = imove;
      moveGraphRev[imove] = 0;
    }

    while (!stack.empty()) {
      auto v = stack.pop_back_val();
      if (!unresolvedNode.contains(v))
        continue;
      auto rev = moveGraphRev[v];
      assert(rev != -1);
      move(v, rev);
      if (moveGraphRev[rev] != -1)
        stack.emplace_back(rev);
    }
  }
}

} // namespace kecc

#endif // KECC_TRANSLATE_MOVE_SCHEDULE_H
