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

  static llvm::DenseSet<LiveRange>
  intersect(const llvm::DenseSet<LiveRange> &a,
            const llvm::DenseSet<LiveRange> &b) {
    llvm::DenseSet<LiveRange> result;
    for (LiveRange elem : a)
      if (b.contains(elem))
        result.insert(elem);
    return result;
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

    auto procInst = [&](ir::InstructionStorage *inst) {
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
    };

    auto *callInst = *I;
    auto start = I;

    for (auto i = ++start, e = end; i != e; ++i) {
      ir::InstructionStorage *inst = *i;
      procInst(inst);
    }

    auto callLiveOut = unionSet(uevar, sub(liveOut, varKill));

    varKill.clear();
    uevar.clear();

    procInst(callInst);

    auto callLiveIn = unionSet(uevar, sub(callLiveOut, varKill));

    end = I;
    liveInMap[callInst] = intersect(callLiveIn, callLiveOut);
    liveOut = callLiveIn;
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

const llvm::DenseSet<LiveRange> &
CallLivenessAnalysis::getLiveIn(ir::InstructionStorage *inst) const {
  assert(inst->hasTrait<ir::CallLike>() && "Instruction must be CallLike");
  return liveIn.at(inst);
}

void CallLivenessAnalysis::dump(llvm::raw_ostream &os) const {
  llvm::SmallVector<ir::InstructionStorage *> insts;
  insts.reserve(liveIn.size());

  for (const auto &[key, _] : liveIn)
    insts.emplace_back(key);

  ir::IRPrintContext printContext(llvm::outs());

  for (ir::Function *func : *getModule()->getIR()) {
    func->registerAllInstInfo(printContext);
  }

  llvm::sort(insts, [&](const auto *l, const auto *r) {
    ir::Function *lFunc = l->getParentBlock()->getParentFunction();
    ir::Function *rFunc = r->getParentBlock()->getParentFunction();
    if (lFunc != rFunc)
      return lFunc->getName() < rFunc->getName();

    auto lResult0 = l->getResult(0);
    auto rResult0 = r->getResult(0);

    auto lRid = printContext.getId(lResult0);
    auto rRid = printContext.getId(rResult0);

    return lRid < rRid;
  });

  LiveRangeAnalysis *liveRangeAnalysis =
      getModule()->getAnalysis<LiveRangeAnalysis>();
  assert(liveRangeAnalysis);

  ir::Function *prevFunc = nullptr;
  os << "Call Liveness dump:\n";
  for (const auto *inst : insts) {
    if (prevFunc != inst->getParentBlock()->getParentFunction()) {
      prevFunc = inst->getParentBlock()->getParentFunction();
      os << "Call instructions in function @" << prevFunc->getName() << ":\n";
    }
    const auto liveInVars = liveIn.at(inst);
    llvm::SmallVector<LiveRange> lives(liveInVars.begin(), liveInVars.end());

    auto currLRToId = liveRangeAnalysis->getFuncLRIdMap(prevFunc);
    llvm::sort(lives, [&](auto lr0, auto lr1) {
      return currLRToId.at(lr0) < currLRToId.at(lr1);
    });

    inst->print(os);
    os << " <-- live in:";
    for (LiveRange lr : lives)
      os << " L" << currLRToId.at(lr);
    os << '\n';
  }
}

} // namespace kecc
