#ifndef KECC_TRANSLATE_LIVE_RANGE_ANALYSIS_H
#define KECC_TRANSLATE_LIVE_RANGE_ANALYSIS_H

#include "kecc/ir/Analysis.h"
#include "kecc/ir/IR.h"
#include "kecc/ir/Instruction.h"
#include "kecc/ir/Value.h"
#include "kecc/translate/LiveRange.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

namespace kecc {
class SpillAnalysis;

struct SpillInfo {
  SpillInfo() = default;
  SpillInfo(llvm::DenseSet<LiveRange> spilled,
            llvm::DenseMap<const ir::Operand *, LiveRange> restore,
            llvm::DenseMap<LiveRange, LiveRange> restoreMemory)
      : spilled(std::move(spilled)), restore(std::move(restore)),
        restoreMemory(std::move(restoreMemory)) {}

  void insert(const SpillInfo &spillInfo) {
    for (auto lr : spillInfo.spilled) {
      auto [it, inserted] = spilled.insert(lr);
      assert(inserted && "Already spillied");
      (void)inserted;
    }

    for (const auto &[op, lr] : spillInfo.restore) {
      auto [it, inserted] = restore.try_emplace(op, lr);
      assert(inserted && "Already restored");
      (void)inserted;
    }

    for (const auto &[op, lr] : spillInfo.restoreMemory) {
      auto [_, inserted] = restoreMemory.try_emplace(op, lr);
      assert(inserted && "Already restored");
      (void)inserted;
    }
  }

  llvm::DenseSet<LiveRange> spilled;
  llvm::DenseMap<const ir::Operand *, LiveRange> restore;
  llvm::DenseMap<LiveRange, LiveRange> restoreMemory;
};

class LiveRangeAnalysisImpl;
class LiveRangeAnalysis : public ir::Analysis {
public:
  ~LiveRangeAnalysis();

  static std::unique_ptr<LiveRangeAnalysis> create(ir::Module *module);

  LiveRange getLiveRange(ir::Value value) const;
  LiveRange getLiveRange(ir::Function *func, ir::Value value) const;

  llvm::ArrayRef<std::pair<LiveRange, LiveRange>>
  getCopyMap(ir::Block *block) const;

  void dump(llvm::raw_ostream &os, const SpillInfo &info = {}) const;

  const llvm::DenseMap<LiveRange, size_t> &
  getFuncLRIdMap(ir::Function *func) const;

  const llvm::DenseMap<ir::Function *, llvm::DenseMap<LiveRange, size_t>> &
  getLRIdMap() const;

  // returns restoring map
  void spill(SpillAnalysis *spillAnalysis, ir::Function *func,
             const llvm::DenseSet<LiveRange> &liveRanges);

private:
  LiveRangeAnalysis(ir::Module *module,
                    std::unique_ptr<LiveRangeAnalysisImpl> impl);

  std::unique_ptr<LiveRangeAnalysisImpl> impl;
};

class LivenessAnalysis : public ir::Analysis {
public:
  static std::unique_ptr<LivenessAnalysis> create(ir::Module *module,
                                                  const SpillInfo &spill = {});

  void dump(llvm::raw_ostream &os) const;

  const llvm::DenseSet<LiveRange> &getLiveVars(ir::Block *block) const;

private:
  LivenessAnalysis(
      ir::Module *module,
      llvm::DenseMap<ir::Block *, llvm::DenseSet<LiveRange>> liveOut)
      : Analysis(module), liveOut(std::move(liveOut)) {}

  const llvm::DenseMap<ir::Block *, llvm::DenseSet<LiveRange>> liveOut;
};

class CallLivenessAnalysis : public ir::Analysis {
public:
  static std::unique_ptr<CallLivenessAnalysis> create(ir::Module *module);

  void dump(llvm::raw_ostream &os) const;

  const llvm::DenseSet<LiveRange> &
  getLiveIn(ir::InstructionStorage *inst) const;

private:
  CallLivenessAnalysis(
      ir::Module *module,
      llvm::DenseMap<ir::InstructionStorage *, llvm::DenseSet<LiveRange>>
          liveIn)
      : Analysis(module), liveIn(std::move(liveIn)) {}
  llvm::DenseMap<ir::InstructionStorage *, llvm::DenseSet<LiveRange>> liveIn;
};

} // namespace kecc

DECLARE_KECC_TYPE_ID(kecc::LiveRangeAnalysis)
DECLARE_KECC_TYPE_ID(kecc::LivenessAnalysis)

#endif // KECC_TRANSLATE_LIVE_RANGE_ANALYSIS_H
