#ifndef KECC_IR_MODULE_H
#define KECC_IR_MODULE_H

#include "kecc/ir/Analysis.h"
#include "kecc/ir/Block.h"
#include "kecc/ir/IR.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseSet.h"

namespace kecc::ir {

using StructFieldsMap =
    llvm::DenseMap<llvm::StringRef, llvm::SmallVector<Type>>;

class Module {
public:
  static std::unique_ptr<Module> create(std::unique_ptr<IR> ir);

  void replaceInst(InstructionStorage *oldInst, InstructionStorage *newInst);
  bool replaceInst(InstructionStorage *oldInst,
                   llvm::function_ref<
                       InstructionStorage *(IRBuilder &, InstructionStorage *)>
                       newInstBuildFunc,
                   bool remove = false);

  void replaceInst(InstructionStorage *oldInst, llvm::ArrayRef<Value> newVals,
                   bool remove = false);

  // Replace a block exit if necessary.
  // returns true if the exit was replaced, false otherwise.
  bool replaceExit(
      BlockExit oldExit,
      llvm::function_ref<BlockExit(IRBuilder &builder, BlockExit oldExit)>
          newExitBuildFunc);

  void removeInst(InstructionStorage *inst);

  IR *getIR() const { return ir.get(); }

  template <typename ConcreteAnalysis> ConcreteAnalysis *getAnalysis() {
    TypeID typeId = TypeID::get<ConcreteAnalysis>();
    auto it = analysisMap.find(typeId);
    if (it != analysisMap.end()) {
      return static_cast<ConcreteAnalysis *>(it->second.get());
    }
    return nullptr;
  }

  template <typename ConcreteAnalysis>
  void insertAnalysis(std::unique_ptr<ConcreteAnalysis> analysis) {
    TypeID typeId = TypeID::get<ConcreteAnalysis>();
    analysisMap[typeId] = std::move(analysis);
  }

  void removeBlock(Block *block);

  const llvm::DenseSet<Block *> &getPredecessors(Block *block) const;
  const llvm::DenseSet<Block *> &getSuccessors(Block *block) const;

  IRContext *getContext() const;

  void updatePredsAndSuccs();

  std::pair<StructSizeMap, StructFieldsMap> calcStructSizeMap() const;

  void addBlockRelation(Block *pred, Block *succ) {
    predecessors[succ].insert(pred);
    successors[pred].insert(succ);
  }

private:
  Module(std::unique_ptr<IR> ir) : ir(std::move(ir)) {}

  std::unique_ptr<IR> ir;
  llvm::DenseMap<Block *, llvm::DenseSet<Block *>> predecessors;
  llvm::DenseMap<Block *, llvm::DenseSet<Block *>> successors;
  llvm::DenseMap<TypeID, std::unique_ptr<Analysis>> analysisMap;
};

} // namespace kecc::ir

#endif // KECC_IR_MODULE_H
