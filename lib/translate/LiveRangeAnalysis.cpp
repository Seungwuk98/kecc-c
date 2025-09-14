#include "kecc/ir/IRAnalyses.h"
#include "kecc/ir/IRInstructions.h"
#include "kecc/ir/Instruction.h"
#include "kecc/translate/LiveRange.h"
#include "kecc/translate/LiveRangeAnalyses.h"
#include "kecc/translate/SpillAnalysis.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

DEFINE_KECC_TYPE_ID(kecc::LiveRangeAnalysis)

namespace kecc {

class LiveRangeAnalysisImpl {
public:
  LiveRangeAnalysisImpl(
      llvm::DenseMap<ir::Function *, llvm::DenseMap<ir::Value, LiveRange>>
          liveRangeMap,
      llvm::DenseMap<ir::Block *,
                     llvm::SmallVector<std::pair<LiveRange, LiveRange>>>
          copyMap,
      std::vector<const LiveRangeStorage *> liveRangeStorages)
      : liveRangeMap(std::move(liveRangeMap)), copyMap(std::move(copyMap)),
        liveRangeStorages(std::move(liveRangeStorages)) {}

  ~LiveRangeAnalysisImpl() {
    for (const LiveRangeStorage *ptr : liveRangeStorages)
      free(const_cast<LiveRangeStorage *>(ptr));
  }

  LiveRange getLiveRange(ir::Function *function, ir::Value value) const;

  LiveRange createNewLiveRange(ir::Function *function, ir::Type type) {
    auto newStorage = new (llvm::safe_malloc(sizeof(LiveRangeStorage)))
        LiveRangeStorage(type, function);
    LiveRange liveRange(newStorage);
    return liveRange;
  }

  void spill(SpillAnalysis *spillAnalysis, ir::Function *func,
             const llvm::DenseSet<LiveRange> &liveRanges);

  llvm::ArrayRef<std::pair<LiveRange, LiveRange>>
  getCopyMap(ir::Block *block) const {
    auto it = copyMap.find(block);
    if (it != copyMap.end())
      return it->second;
    return {};
  }

  void updateLiveRangeIdMap(ir::Module *module, ir::Function *func);

private:
  friend class LiveRangeAnalysis;
  llvm::DenseMap<ir::Function *, llvm::DenseMap<ir::Value, LiveRange>>
      liveRangeMap;
  llvm::DenseMap<ir::Block *,
                 llvm::SmallVector<std::pair<LiveRange, LiveRange>>>
      copyMap;
  std::vector<const LiveRangeStorage *> liveRangeStorages;

  llvm::DenseMap<ir::Function *, llvm::DenseMap<LiveRange, size_t>>
      liveRangeFuncIdMap;
};

void LiveRangeAnalysisImpl::spill(
    SpillAnalysis *spillAnalysis, ir::Function *func,
    const llvm::DenseSet<LiveRange> &spilledLiveRanges) {
  assert(func->hasDefinition() && "Function should have a definition");
  auto &lrMap = liveRangeMap[func];

  // (to, from) LiveRange `to` is must be restored from `from`'s spill memory.
  llvm::DenseMap<LiveRange, LiveRange> restoreMemoryMap;
  llvm::DenseMap<const ir::Operand *, LiveRange> restoreMap;
  for (ir::Block *block : *func) {
    for (ir::InstructionStorage *inst : *block) {
      llvm::ArrayRef<ir::Operand> operands;
      if (inst->hasTrait<ir::CallLike>()) {
        if (auto call = inst->getDefiningInst<ir::inst::Call>())
          operands = call.getFunctionAsOperand();
      } else {
        operands = inst->getOperands();
      }
      for (const ir::Operand &operand : operands) {
        if (operand.isConstant())
          continue;

        auto liveRange = lrMap[operand];
        if (spilledLiveRanges.contains(liveRange)) {
          auto newLiveRange = createNewLiveRange(func, operand.getType());
          restoreMemoryMap[newLiveRange] = liveRange;
          restoreMap[&operand] = newLiveRange;
        }
      }
    }

    for (auto &[to, from] : copyMap[block]) {
      if (spilledLiveRanges.contains(from)) {
        auto newLiveRange = createNewLiveRange(func, from.getType());
        restoreMemoryMap[newLiveRange] = from;
        from = newLiveRange;
      }
    }
  }

  spillAnalysis->insertSpillInfo(
      {spilledLiveRanges, restoreMap, restoreMemoryMap});

  updateLiveRangeIdMap(spillAnalysis->getModule(), func);
}

void LiveRangeAnalysisImpl::updateLiveRangeIdMap(ir::Module *module,
                                                 ir::Function *func) {
  llvm::DenseMap<LiveRange, size_t> &liveRangeIdMap = liveRangeFuncIdMap[func];
  liveRangeIdMap.clear();
  assert(func->hasDefinition() && "Function should have a definition");
  size_t id = 0;

  auto *spillAnalysis = module->getAnalysis<SpillAnalysis>();
  SpillInfo spillInfo =
      spillAnalysis ? spillAnalysis->getSpillInfo() : SpillInfo();

  for (ir::InstructionStorage *inst : *func->getAllocationBlock()) {
    auto localVar = inst->getDefiningInst<ir::inst::LocalVariable>();
    assert(localVar);
    auto lr = getLiveRange(func, localVar);
    assert(!liveRangeIdMap.contains(lr) && "Live range already exists");
    liveRangeIdMap[lr] = id++;
  }

  for (ir::Block *block : *func) {
    for (auto I = block->begin(), E = block->end(); I != E; ++I) {
      ir::InstructionStorage *inst = *I;

      if (inst->getDefiningInst<ir::BlockExit>()) {
        for (const auto &[to, from] : getCopyMap(block)) {
          // `to`'s id must be counted in successor's block
          if (!liveRangeIdMap.contains(from))
            liveRangeIdMap[from] = id++;
        }
      }

      for (const ir::Operand &operand : inst->getOperands()) {
        if (operand.isConstant())
          continue;

        if (auto it = spillInfo.restore.find(&operand);
            it != spillInfo.restore.end()) {
          auto lr = it->getSecond();
          if (liveRangeIdMap.contains(lr))
            continue;
          liveRangeIdMap[lr] = id++;
        }
      }

      auto results = inst->getResults();
      for (ir::Value result : results) {
        auto lr = getLiveRange(func, result);
        if (liveRangeIdMap.contains(lr))
          continue;
        liveRangeIdMap[lr] = id++;
      }
    }
  }
}

struct LiveRangeAnalysisBuilder {
  LiveRangeAnalysisBuilder(ir::Module *module, ir::Function *function,
                           llvm::ArrayRef<ir::Block *> order)
      : function(function), rpo(order) {
    build();
  }

  LiveRange createLiveRange(ir::Function *func, ir::Type type) {
    auto newStorage = new (llvm::safe_malloc(sizeof(LiveRangeStorage)))
        LiveRangeStorage(type, func);
    LiveRange liveRange(newStorage);
    liveRangeStorages.emplace_back(newStorage);
    return liveRange;
  }

  void insertLiveRange(ir::Value value) {
    auto it = liveRangeMap.find(value);
    assert(it == liveRangeMap.end() && "Live range already exists");
    auto newLiveRange = createLiveRange(function, value.getType());
    liveRangeMap[value] = newLiveRange;
    liveRangeTypeMap[newLiveRange] = value.getType();
  }

  bool hasLiveRange(ir::Value value) const {
    return liveRangeMap.contains(value);
  }

  // (to, from)
  llvm::DenseMap<ir::Block *,
                 llvm::SmallVector<std::pair<LiveRange, LiveRange>>>
      copyMap;
  llvm::DenseMap<ir::Value, LiveRange> liveRangeMap;
  llvm::DenseMap<LiveRange, ir::Type> liveRangeTypeMap;

  std::vector<const LiveRangeStorage *> liveRangeStorages;

  ir::Module *module;
  ir::Function *function;
  llvm::ArrayRef<ir::Block *> rpo;

private:
  void build();
};

LiveRangeAnalysis::LiveRangeAnalysis(
    ir::Module *module, std::unique_ptr<LiveRangeAnalysisImpl> impl)
    : Analysis(module), impl(std::move(impl)) {}
LiveRangeAnalysis::~LiveRangeAnalysis() = default;

LiveRange LiveRangeAnalysis::getLiveRange(ir::Value value) const {
  return impl->getLiveRange(
      value.getInstruction()->getParentBlock()->getParentFunction(), value);
}

LiveRange LiveRangeAnalysis::getLiveRange(ir::Function *function,
                                          ir::Value value) const {
  return impl->getLiveRange(function, value);
}

std::unique_ptr<LiveRangeAnalysis>
LiveRangeAnalysis::create(ir::Module *module) {
  auto ir = module->getIR();
  auto orderAnalysis = module->getAnalysis<ir::VisitOrderAnalysis>();
  if (!orderAnalysis) {
    auto visitOrder = ir::VisitOrderAnalysis::create(module);
    module->insertAnalysis(std::move(visitOrder));
    orderAnalysis = module->getAnalysis<ir::VisitOrderAnalysis>();
  }

  llvm::DenseMap<ir::Function *, llvm::DenseMap<ir::Value, LiveRange>>
      liveRangeMap;
  llvm::DenseMap<ir::Block *,
                 llvm::SmallVector<std::pair<LiveRange, LiveRange>>>
      copyMap;
  std::vector<const LiveRangeStorage *> liveRangeStorage;
  for (ir::Function *func : *module->getIR()) {
    if (!func->hasDefinition())
      continue;

    auto rpo = orderAnalysis->getReversePostOrder(func);
    LiveRangeAnalysisBuilder builder(module, func, rpo);
    liveRangeMap[func] = std::move(builder.liveRangeMap);
    liveRangeStorage.insert(liveRangeStorage.end(),
                            builder.liveRangeStorages.begin(),
                            builder.liveRangeStorages.end());
    copyMap.insert(builder.copyMap.begin(), builder.copyMap.end());
  }

  auto impl = std::make_unique<LiveRangeAnalysisImpl>(
      std::move(liveRangeMap), std::move(copyMap), std::move(liveRangeStorage));

  for (ir::Function *func : *module->getIR()) {
    if (!func->hasDefinition())
      continue;
    impl->updateLiveRangeIdMap(module, func);
  }

  return std::unique_ptr<LiveRangeAnalysis>(
      new LiveRangeAnalysis(module, std::move(impl)));
}

void LiveRangeAnalysisBuilder::build() {
  for (ir::InstructionStorage *inst : *function->getAllocationBlock()) {
    auto localVar = inst->getDefiningInst<ir::inst::LocalVariable>();
    assert(localVar && "Expected local variable instruction");
    insertLiveRange(localVar);
  }

  for (ir::Block *block : rpo) {
    for (ir::InstructionStorage *inst : *block) {
      auto results = inst->getResults();

      for (ir::Value result : results)
        insertLiveRange(result);
    }
  }

  for (ir::Block *block : rpo) {
    auto phiIdx = 0;

    auto exit = block->getExit();
    for (ir::JumpArg *jumpArg : exit->getJumpArgs()) {
      auto succ = jumpArg->getBlock();

      auto phiIdx = 0;
      for (auto I = succ->phiBegin(), E = succ->phiEnd(); I != E;
           ++I, ++phiIdx) {
        auto phi = (*I)->getDefiningInst<ir::Phi>();
        assert(phi && "Expected phi instruction");
        ir::Value arg = jumpArg->getArgs()[phiIdx];

        assert(!arg.isConstant() &&
               "Expected non-constant argument for phi instruction");

        auto phiLR = liveRangeMap.at(phi);
        auto argLR = liveRangeMap.at(arg);
        copyMap[block].emplace_back(phiLR, argLR);
      }
    }
  }
}

LiveRange LiveRangeAnalysisImpl::getLiveRange(ir::Function *function,
                                              ir::Value value) const {
  auto it = liveRangeMap.find(function);
  assert(it != liveRangeMap.end() && "Function should have a live range map");
  auto &map = it->second;
  return map.at(value);
}

namespace print {
using namespace kecc::ir::inst;
using Operand = kecc::ir::Operand;

class LiveRangePrinter : public ir::ValuePrinter {
public:
  LiveRangePrinter(
      const SpillInfo &spillInfo, const LiveRangeAnalysis *liveRangeAnalysis,
      const llvm::DenseMap<ir::Function *, llvm::DenseMap<LiveRange, size_t>>
          &liveRangeToId)
      : spillInfo(spillInfo), liveRangeAnalysis(liveRangeAnalysis),
        liveRangeToId((liveRangeToId)) {}

  void printValue(ir::Value value, ir::IRPrintContext &context,
                  bool printName) override {
    if (auto constant = value.getDefiningInst<ir::inst::Constant>()) {
      constant->print(context);
      return;
    }

    auto lr = liveRangeAnalysis->getLiveRange(
        value.getInstruction()->getParentBlock()->getParentFunction(), value);
    size_t id = liveRangeToId.at(lr.getFunction()).at(lr);
    context.getOS() << "L" << id << ":" << lr.getType();
    if (printName && !value.getValueName().empty())
      context.getOS() << ":" << value.getValueName();
  }

  void printOperand(const Operand &operand,
                    ir::IRPrintContext &context) override {
    if (auto constant = operand.getDefiningInst<ir::inst::Constant>()) {
      constant->print(context);
      return;
    }

    LiveRange lr;
    if (auto it = spillInfo.restore.find(&operand);
        it != spillInfo.restore.end()) {
      lr = it->getSecond();
    } else {
      lr = liveRangeAnalysis->getLiveRange(operand);
    }

    size_t id = liveRangeToId.at(lr.getFunction()).at(lr);
    context.getOS() << 'L' << id << ':' << lr.getType();
  }

  void printJumpArg(const ir::JumpArg *jumpArg,
                    ir::IRPrintContext &context) override {
    context.getOS() << "b" << jumpArg->getBlock()->getId();
  }

private:
  const SpillInfo &spillInfo;
  const LiveRangeAnalysis *liveRangeAnalysis;
  const llvm::DenseMap<ir::Function *, llvm::DenseMap<LiveRange, size_t>>
      &liveRangeToId;
};

} // namespace print

void LiveRangeAnalysis::dump(llvm::raw_ostream &os,
                             const SpillInfo &spillInfo) const {
  os << "Live Range Analysis dump:\n";

  auto printIndent = [&os](size_t indent) {
    for (size_t i = 0; i < indent; ++i)
      os << ' ';
  };

  ir::IRPrintContext printContext(os, std::make_unique<print::LiveRangePrinter>(
                                          spillInfo, this, getLRIdMap()));
  ir::IRPrintContext::AddIndent addIndent(printContext);

  bool first = true;
  size_t indent = 0;
  for (ir::Function *func : *getModule()->getIR()) {
    if (first)
      first = false;
    else
      os << "\n";
    ir::FunctionT funcT = func->getFunctionType().cast<ir::FunctionT>();
    auto retTypes = funcT.getReturnTypes();
    auto argTypes = funcT.getArgTypes();

    os << "fun ";
    for (size_t i = 0; i < retTypes.size(); ++i) {
      if (i > 0)
        os << ", ";
      os << retTypes[i];
    }

    os << " @" << func->getName() << " (";
    for (size_t i = 0; i < argTypes.size(); ++i) {
      if (i > 0)
        os << ", ";
      os << argTypes[i];
    }

    os << ")";
    if (!func->hasDefinition()) {
      os << '\n';
      continue;
    }

    llvm::DenseMap<LiveRange, size_t> liveRangeToId = getFuncLRIdMap(func);
    auto printSpill = [&](llvm::ArrayRef<ir::Value> results, size_t indent) {
      llvm::SmallVector<LiveRange> spills;
      llvm::for_each(results, [&](ir::Value value) {
        auto lr = getLiveRange(value);
        if (spillInfo.spilled.contains(lr))
          spills.emplace_back(lr);
      });
      if (spills.empty())
        return;

      printIndent(indent);
      os << "spill ";
      for (size_t i = 0; i < spills.size(); ++i) {
        if (i > 0)
          os << ", ";
        os << 'L' << liveRangeToId.at(spills[i]);
      }
      os << '\n';
    };

    auto printRestore = [&](llvm::ArrayRef<ir::Operand> operands,
                            size_t indent) {
      for (const ir::Operand &operand : operands) {
        if (auto it = spillInfo.restore.find(&operand);
            it != spillInfo.restore.end()) {
          auto originalLR = getLiveRange(operand);
          auto newLR = it->getSecond();
          printIndent(indent);
          os << "restore L" << liveRangeToId.at(newLR)
             << " from spill memory of L" << liveRangeToId.at(originalLR)
             << '\n';
        }
      }
    };

    os << " {\n";

    os << "init:\n";
    printIndent(indent + 2);
    os << "bid: b" << func->getEntryBlock()->getId() << '\n';

    printIndent(indent + 2);
    os << "allocations:\n";
    {
      auto allocIdx = 0;
      for (ir::InstructionStorage *inst : *func->getAllocationBlock()) {
        auto localVar = inst->getDefiningInst<ir::inst::LocalVariable>();
        assert(localVar && "Expected local variable instruction");
        printIndent(indent + 4);

        os << "L" << liveRangeToId.at(getLiveRange(func, localVar)) << ":"
           << localVar.getType().cast<ir::PointerT>().getPointeeType() << '\n';
        printSpill(localVar.getResult(), indent + 4);
      }
    }

    for (ir::Block *block : *func) {
      os << "\nblock b" << block->getId() << ":\n";

      for (auto I = block->tempBegin(), E = block->tempEnd(); I != E; ++I) {
        auto inst = *I;
        if (!inst->hasTrait<ir::CallLike>())
          printRestore(inst->getOperands(), indent + 2);

        printIndent(indent + 2);

        inst->print(printContext);

        os << '\n';
        auto results = inst->getResults();
        printSpill(results, indent + 2);
      }

      auto exit = block->getExit();
      printRestore(exit->getOperands(), indent + 2);
      for (const auto &[to, from] : impl->getCopyMap(block)) {
        // if `to` were spilled, store `from` into `to`'s spill memory
        // if `from` were restored, restore `from` into `to`
        auto toSpilled = spillInfo.spilled.contains(to);
        auto fromRestored = spillInfo.restoreMemory.contains(from);
        if (toSpilled && !fromRestored) {
          printIndent(indent + 2);
          os << "store L" << liveRangeToId.at(from) << " into spill memory of L"
             << liveRangeToId.at(to) << '\n';
        } else if (fromRestored) {
          printIndent(indent + 2);
          auto restoreMemory = spillInfo.restoreMemory.at(from);

          if (!toSpilled) {
            os << 'L' << liveRangeToId.at(to)
               << " = restore from spill memory of L"
               << liveRangeToId.at(restoreMemory) << '\n';
          } else {
            os << "memcpy to spill memory of L" << liveRangeToId.at(to)
               << " from spill memory of L" << liveRangeToId.at(restoreMemory)
               << '\n';
          }
        } else {
          printIndent(indent + 2);
          os << 'L' << liveRangeToId.at(to) << " = L" << liveRangeToId.at(from)
             << '\n';
        }
      }

      printIndent(indent + 2);
      exit->print(printContext);
      os << '\n';
#undef CASE
    }

    os << "}\n";
  }
}

const llvm::DenseMap<LiveRange, size_t> &
LiveRangeAnalysis::getFuncLRIdMap(ir::Function *func) const {
  return impl->liveRangeFuncIdMap.at(func);
}

const llvm::DenseMap<ir::Function *, llvm::DenseMap<LiveRange, size_t>> &
LiveRangeAnalysis::getLRIdMap() const {
  return impl->liveRangeFuncIdMap;
}

llvm::ArrayRef<std::pair<LiveRange, LiveRange>>
LiveRangeAnalysis::getCopyMap(ir::Block *block) const {
  return impl->getCopyMap(block);
}

void LiveRangeAnalysis::spill(SpillAnalysis *spillAnalysis, ir::Function *func,
                              const llvm::DenseSet<LiveRange> &liveRanges) {
  impl->spill(spillAnalysis, func, liveRanges);
}

} // namespace kecc
