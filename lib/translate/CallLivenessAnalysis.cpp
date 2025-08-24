#include "kecc/ir/Instruction.h"
#include "kecc/ir/Module.h"
#include "kecc/translate/LiveRangeAnalyses.h"
#include "kecc/translate/SpillAnalysis.h"

namespace kecc {

namespace {

static llvm::DenseSet<LiveRange> sub(const llvm::DenseSet<LiveRange> &a,
                                     const llvm::DenseSet<LiveRange> &b) {
  llvm::DenseSet<LiveRange> result;
  for (LiveRange elem : a)
    if (!b.contains(elem))
      result.insert(elem);
  return result;
}

static llvm::DenseSet<LiveRange> unionSet(const llvm::DenseSet<LiveRange> &a,
                                          const llvm::DenseSet<LiveRange> &b) {
  llvm::DenseSet<LiveRange> result = a;
  result.insert(b.begin(), b.end());
  return result;
}

} // namespace

class CallLivenessAnalysisBuilder {
public:
  CallLivenessAnalysisBuilder(ir::Block *block,
                              LivenessAnalysis *livenessAnalysis,
                              LiveRangeAnalysis *liveRangeAnalysis,
                              const SpillInfo &spill)
      : block(block), livenessAnalysis(livenessAnalysis),
        liveRangeAnalysis(liveRangeAnalysis), spill(spill) {}

  void build();

  const llvm::DenseMap<ir::InstructionStorage *, llvm::DenseSet<LiveRange>> &
  getLiveInMap() const {
    return liveInMap;
  }

private:
  ir::Block *block;
  LivenessAnalysis *livenessAnalysis;
  LiveRangeAnalysis *liveRangeAnalysis;
  llvm::DenseMap<ir::InstructionStorage *, llvm::DenseSet<LiveRange>> liveInMap;
  const SpillInfo &spill;
};

void CallLivenessAnalysisBuilder::build() {
  llvm::SmallVector<decltype(block->begin())> instructions;
  ir::Function *func = block->getParentFunction();

  for (auto I = block->tempBegin(), E = block->tempEnd(); I != E; ++I) {
    ir::InstructionStorage *inst = *I;
    if (inst->hasTrait<ir::CallLike>())
      instructions.emplace_back(I);
  }

  auto liveOut = livenessAnalysis->getLiveVars(block);
  auto end = block->end();
  for (auto I : llvm::reverse(instructions)) {
    llvm::DenseSet<LiveRange> varKill;
    llvm::DenseSet<LiveRange> uevar;

    for (auto i = I, e = end; i != e; ++i) {
      ir::InstructionStorage *inst = *i;

      if (inst->getDefiningInst<ir::BlockExit>()) {
        for (const auto &[to, from] : liveRangeAnalysis->getCopyMap(block)) {
          if (spill.restoreMemory.contains(from))
            varKill.insert(from);
          else if (!varKill.contains(from))
            uevar.insert(from);

          // if `to` is a spilled value, this copy does not affect the
          // register
          if (!spill.spilled.contains(to))
            varKill.insert(to);
        }
      }

      auto results = inst->getResults();

      for (const ir::Operand &operand : inst->getOperands()) {
        if (operand.isConstant())
          continue;

        // If the operand is a restored value, we need to insert it into
        // varKill
        LiveRange liveRange;
        if (auto it = spill.restore.find(&operand); it != spill.restore.end()) {
          liveRange = it->getSecond();
          varKill.insert(liveRange);
        } else {
          liveRange = liveRangeAnalysis->getLiveRange(func, operand);
          if (spill.spilled.contains(liveRange)) {
            assert((inst->hasTrait<ir::CallLike>()) && "This case is only "
                                                       "possible for call-like "
                                                       "instructions");
            continue;
          }
        }

        if (!varKill.contains(liveRange))
          uevar.insert(liveRange);
      }

      for (ir::Value result : results) {
        LiveRange liveRange = liveRangeAnalysis->getLiveRange(func, result);
        varKill.insert(liveRange);
      }
      // We need to handle spill values but the values are already inserted
      // in varKill, so we can skip them here.
    }

    liveOut = unionSet(uevar, sub(liveOut, varKill));
    end = I;
    liveInMap[*I] = liveOut;
  }
}

std::unique_ptr<CallLivenessAnalysis>
CallLivenessAnalysis::create(ir::Module *module) {
  auto *livenessAnalysis = module->getAnalysis<LivenessAnalysis>();
  auto *liveRangeAnalysis = module->getAnalysis<LiveRangeAnalysis>();
  assert(livenessAnalysis && liveRangeAnalysis &&
         "Liveness and LiveRange analyses must be available");
  auto *spillAnalysis = module->getAnalysis<SpillAnalysis>();

  llvm::DenseMap<ir::InstructionStorage *, llvm::DenseSet<LiveRange>> liveIn;
  for (ir::Function *func : *module->getIR()) {
    if (!func->hasDefinition())
      continue;

    for (ir::Block *block : *func) {
      CallLivenessAnalysisBuilder builder(
          block, livenessAnalysis, liveRangeAnalysis,
          spillAnalysis ? spillAnalysis->getSpillInfo() : SpillInfo{});
      builder.build();

      auto liveInMap = builder.getLiveInMap();
      liveIn.insert(liveInMap.begin(), liveInMap.end());
    }
  }

  return std::unique_ptr<CallLivenessAnalysis>(
      new CallLivenessAnalysis(module, std::move(liveIn)));
}

} // namespace kecc
