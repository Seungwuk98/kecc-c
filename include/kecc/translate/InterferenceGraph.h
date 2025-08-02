#ifndef KECC_TRANSLATE_INTERFERENCE_GRAPH_H
#define KECC_TRANSLATE_INTERFERENCE_GRAPH_H

#include "kecc/ir/IR.h"
#include "kecc/ir/Module.h"
#include <map>
#include <set>

namespace kecc {

class InterferenceGraph {
public:
  static std::unique_ptr<InterferenceGraph>
  create(ir::Module *module, ir::Function *func, bool isFloat);

private:
  InterferenceGraph(std::map<size_t, std::set<size_t>> graph)
      : graph(std::move(graph)) {}

  std::map<size_t, std::set<size_t>> graph;
};

} // namespace kecc

#endif // KECC_TRANSLATE_INTERFERENCE_GRAPH_H
