#include "kecc/ir/IRAnalyses.h"
#include "kecc/ir/IRInstructions.h"
#include "kecc/ir/Instruction.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

DEFINE_KECC_TYPE_ID(kecc::ir::LiveRangeAnalysis)

namespace kecc::ir {

class LiveRangeAnalysisImpl {
public:
  LiveRangeAnalysisImpl(
      llvm::DenseMap<Function *, llvm::DenseMap<Value, size_t>> liveRangeMap,
      llvm::DenseMap<Function *, llvm::DenseMap<size_t, Type>> liveRangeTypeMap)
      : liveRangeMap(std::move(liveRangeMap)),
        liveRangeTypeMap(std::move(liveRangeTypeMap)) {}

  size_t getLiveRange(Function *function, Value value) const;
  Type getLiveRangeType(Function *function, size_t liveRange) const;

private:
  llvm::DenseMap<Function *, llvm::DenseMap<Value, size_t>> liveRangeMap;
  llvm::DenseMap<Function *, llvm::DenseMap<size_t, Type>> liveRangeTypeMap;
};

struct LiveRangeAnalysisBuilder {
  LiveRangeAnalysisBuilder(Module *module, Function *function,
                           llvm::ArrayRef<Block *> order)
      : function(function), rpo(order) {
    build();
  }

  Value find(Value v) {
    Value &par = parent[v];
    if (!par)
      return par = v;

    if (par == v)
      return v;

    return par = find(par);
  }

  size_t getRank(Value v) {
    if (auto it = rank.find(v); it != rank.end())
      return it->second;
    return rank[v] = 0;
  }

  void union_(Value a, Value b) {
    a = find(a);
    b = find(b);

    if (a == b)
      return;

    auto rankA = getRank(a);
    auto rankB = getRank(b);

    if (rankA < rankB)
      parent[a] = b;
    else if (rankA > rankB)
      parent[b] = a;
    else {
      parent[b] = a;
      rank[a]++;
    }
  }

  void insertLiveRange(Value value, size_t liveRange) {
    auto it = liveRangeMap.find(value);
    assert(it == liveRangeMap.end() && "Live range already exists");
    liveRangeMap[value] = liveRange;
    liveRangeTypeMap[liveRange] = value.getType();
  }

  bool hasLiveRange(Value value) const { return liveRangeMap.contains(value); }

  llvm::DenseMap<Value, Value> parent;
  llvm::DenseMap<Value, size_t> rank;

  llvm::DenseMap<Value, size_t> liveRangeMap;
  llvm::DenseMap<size_t, Type> liveRangeTypeMap;

  Module *module;
  Function *function;
  llvm::ArrayRef<Block *> rpo;

private:
  void build();
};

LiveRangeAnalysis::LiveRangeAnalysis(
    Module *module, std::unique_ptr<LiveRangeAnalysisImpl> impl)
    : Analysis(module), impl(std::move(impl)) {}
LiveRangeAnalysis::~LiveRangeAnalysis() = default;

size_t LiveRangeAnalysis::getLiveRange(Function *function, Value value) const {
  return impl->getLiveRange(function, value);
}
Type LiveRangeAnalysis::getLiveRangeType(Function *function,
                                         size_t liveRange) const {
  return impl->getLiveRangeType(function, liveRange);
}

std::unique_ptr<LiveRangeAnalysis> LiveRangeAnalysis::create(Module *module) {
  auto ir = module->getIR();
  auto orderAnalysis = module->getAnalysis<VisitOrderAnalysis>();
  if (!orderAnalysis) {
    auto visitOrder = VisitOrderAnalysis::create(module);
    module->insertAnalysis(std::move(visitOrder));
    orderAnalysis = module->getAnalysis<VisitOrderAnalysis>();
  }

  llvm::DenseMap<Function *, llvm::DenseMap<Value, size_t>> liveRangeMap;
  llvm::DenseMap<Function *, llvm::DenseMap<size_t, Type>> liveRangeTypeMap;
  for (Function *func : *module->getIR()) {
    if (!func->hasDefinition())
      continue;

    auto rpo = orderAnalysis->getReversePostOrder(func);
    LiveRangeAnalysisBuilder builder(module, func, rpo);
    liveRangeMap[func] = std::move(builder.liveRangeMap);
    liveRangeTypeMap[func] = std::move(builder.liveRangeTypeMap);
  }

  auto impl = std::make_unique<LiveRangeAnalysisImpl>(
      std::move(liveRangeMap), std::move(liveRangeTypeMap));

  return std::unique_ptr<LiveRangeAnalysis>(
      new LiveRangeAnalysis(module, std::move(impl)));
}

void LiveRangeAnalysisBuilder::build() {
  for (Block *block : rpo) {
    auto exit = block->getExit();
    for (JumpArg *jumpArg : exit->getJumpArgs()) {
      auto succ = jumpArg->getBlock();
      auto succPhiIdx = 0;
      for (auto phiBegin = succ->phiBegin(), phiEnd = succ->phiEnd();
           phiBegin != phiEnd; ++phiBegin, ++succPhiIdx) {
        auto phi = (*phiBegin)->getDefiningInst<Phi>();
        assert(phi && "Expected Phi instruction in successor block");
        Value arg = jumpArg->getArgs()[succPhiIdx];
        union_(phi, jumpArg->getArgs()[succPhiIdx]);
      }
      assert(jumpArg->getArgs().size() == succPhiIdx &&
             "Jump argument size mismatch with successor Phi arguments");
    }
  }

  size_t liveRange = 0;
  for (InstructionStorage *inst : *function->getAllocationBlock()) {
    auto localVar = inst->getDefiningInst<inst::LocalVariable>();
    assert(localVar && "Expected local variable instruction");
    insertLiveRange(localVar, liveRange++);
  }

  for (Block *block : rpo) {
    for (InstructionStorage *inst : *block) {
      auto results = inst->getResults();

      for (Value result : results) {
        auto root = find(result);
        if (!hasLiveRange(root))
          insertLiveRange(root, liveRange++);
        if (result != root)
          insertLiveRange(result, liveRangeMap[root]);
      }
    }
  }
}

size_t LiveRangeAnalysisImpl::getLiveRange(Function *function,
                                           Value value) const {
  auto it = liveRangeMap.find(function);
  assert(it != liveRangeMap.end() && "Function should have a live range map");
  auto &map = it->second;
  return map.at(value);
}

Type LiveRangeAnalysisImpl::getLiveRangeType(Function *function,
                                             size_t liveRange) const {
  auto it = liveRangeTypeMap.find(function);
  assert(it != liveRangeTypeMap.end() &&
         "Function should have a live range type map");
  auto &map = it->second;
  return map.at(liveRange);
}

namespace print {
using namespace kecc::ir::inst;

#define PRINT_FUNC(Inst)                                                       \
  static void print##Inst(Inst inst, llvm::raw_ostream &os,                    \
                          llvm::function_ref<void(Value)> printValue,          \
                          size_t indent,                                       \
                          llvm::function_ref<void(size_t)> printIndent)

PRINT_FUNC(Nop) { os << " = nop"; }
PRINT_FUNC(Load) {
  os << " = load ";
  printValue(inst.getPointer());
}
PRINT_FUNC(Store) {
  os << " = store ";
  printValue(inst.getValue());
  os << ' ';
  printValue(inst.getPointer());
}
PRINT_FUNC(Call) {
  os << "call ";
  printValue(inst.getFunction());
  os << "(";
  for (size_t i = 0; i < inst.getArguments().size(); ++i) {
    if (i > 0)
      os << ", ";
    printValue(inst.getArguments()[i]);
  }
  os << ")";
}
PRINT_FUNC(TypeCast) {
  os << " = typecast ";
  printValue(inst.getValue());
  os << " to " << inst.getType();
}
PRINT_FUNC(Gep) {
  os << " = getelementptr ";
  printValue(inst.getBasePointer());
  os << " offset ";
  printValue(inst.getOffset());
}
PRINT_FUNC(Binary) {
  os << " = ";
  switch (inst.getOpKind()) {
  case Binary::Add:
    os << "add";
    break;
  case Binary::Sub:
    os << "sub";
    break;
  case Binary::Mul:
    os << "mul";
    break;
  case Binary::Div:
    os << "div";
    break;
  case Binary::Mod:
    os << "mod";
    break;
  case Binary::BitAnd:
    os << "and";
    break;
  case Binary::BitOr:
    os << "or";
    break;
  case Binary::BitXor:
    os << "xor";
    break;
  case Binary::Shl:
    os << "shl";
    break;
  case Binary::Shr:
    os << "shr";
    break;
  case Binary::Eq:
    os << "cmp eq";
    break;
  case Binary::Ne:
    os << "cmp ne";
    break;
  case Binary::Lt:
    os << "cmp lt";
    break;
  case Binary::Le:
    os << "cmp le";
    break;
  case Binary::Gt:
    os << "cmp gt";
    break;
  case Binary::Ge:
    os << "cmp ge";
    break;
  }

  os << ' ';
  printValue(inst.getLhs());
  os << ' ';
  printValue(inst.getRhs());
}
PRINT_FUNC(Unary) {
  os << " = ";
  switch (inst.getOpKind()) {
  case Unary::Plus:
    os << "plus";
    break;
  case Unary::Minus:
    os << "minus";
    break;
  case Unary::Negate:
    os << "negate";
    break;
  }
  os << ' ';
  printValue(inst.getValue());
}
PRINT_FUNC(Jump) { os << "j b" << inst.getJumpArg()->getBlock()->getId(); }
PRINT_FUNC(Branch) {
  os << "br ";
  printValue(inst.getCondition());
  os << ", b" << inst.getIfArg()->getBlock()->getId() << ", "
     << "b" << inst.getElseArg()->getBlock()->getId();
}
PRINT_FUNC(Switch) {
  os << "switch ";
  printValue(inst.getValue());
  os << " default b" << inst.getDefaultCase()->getBlock()->getId() << " [";
  for (auto idx = 0u; idx < inst.getCaseSize(); ++idx) {
    os << '\n';
    printIndent(indent + 2);
    printValue(inst.getCaseValue(idx));
    os << " b" << inst.getCaseJumpArg(idx)->getBlock()->getId();
  }
  os << '\n';
  printIndent(indent);
  os << "]";
}
PRINT_FUNC(Return) {
  os << "ret ";
  for (size_t i = 0; i < inst.getValues().size(); ++i) {
    if (i > 0)
      os << ", ";
    printValue(inst.getValues()[i]);
  }
}
PRINT_FUNC(Unreachable) { os << "unreachable"; }
PRINT_FUNC(OutlineConstant) {
  os << " = outline constant ";
  printValue(inst.getConstant());
}
PRINT_FUNC(InlineCall) {
  os << " = inline call @" << inst.getName() << ':' << inst.getFunctionType()
     << ')';
  for (size_t i = 0; i < inst.getArguments().size(); ++i) {
    if (i > 0)
      os << ", ";
    printValue(inst.getArguments()[i]);
  }
  os << ')';
}

#undef PRINT_FUNC

} // namespace print

void LiveRangeAnalysis::dump(llvm::raw_ostream &os) const {
  os << "Live Range Analysis dump:\n";

  auto printIndent = [&os](size_t indent) {
    for (size_t i = 0; i < indent; ++i)
      os << ' ';
  };

  auto printValue = [&](Value value) {
    if (auto constant = value.getDefiningInst<inst::Constant>()) {
      IRPrintContext context(os);
      constant->print(context);
      return;
    }
    os << "L"
       << getLiveRange(
              value.getInstruction()->getParentBlock()->getParentFunction(),
              value)
       << ":" << value.getType();
    if (!value.getValueName().empty())
      os << ":" << value.getValueName();
  };

  bool first = true;
  size_t indent = 0;
  for (Function *func : *getModule()->getIR()) {
    if (first)
      first = false;
    else
      os << "\n";
    FunctionT funcT = func->getFunctionType().cast<FunctionT>();
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

    os << " {\n";

    os << "init:\n";
    printIndent(indent + 2);
    os << "bid: b" << func->getEntryBlock()->getId() << '\n';

    printIndent(indent + 2);
    os << "allocations:\n";
    {
      auto allocIdx = 0;
      for (InstructionStorage *inst : *func->getAllocationBlock()) {
        auto localVar = inst->getDefiningInst<inst::LocalVariable>();
        assert(localVar && "Expected local variable instruction");
        printIndent(indent + 4);

        os << "L" << getLiveRange(func, localVar) << ":"
           << localVar.getType().cast<PointerT>().getPointeeType() << '\n';
      }
    }

    for (Block *block : *func) {
      os << "\nblock b" << block->getId() << ":\n";

      for (InstructionStorage *inst : *block) {
        printIndent(indent + 2);
        auto results = inst->getResults();
        for (size_t i = 0; i < results.size(); ++i) {
          if (i > 0)
            os << ", ";
          printValue(results[i]);
        }

        llvm::TypeSwitch<Instruction, void>(Instruction(inst))
            .Case([&](Phi) {})
#define CASE(Inst)                                                             \
  .Case([&](inst::Inst inst) {                                                 \
    print::print##Inst(inst, os, printValue, indent + 2, printIndent);         \
  })
            // clang-format off
              CASE(Nop)
              CASE(Load)
              CASE(Store)
              CASE(Call) 
              CASE(TypeCast)
              CASE(Gep)
              CASE(Binary)
              CASE(Unary)
              CASE(Jump)
              CASE(Branch)
              CASE(Switch)
              CASE(Return) 
              CASE(Unreachable)
              CASE(OutlineConstant)
              CASE(InlineCall)
#undef CASE
              .Default([&](Instruction inst) {
                llvm_unreachable("Can't dump the instruction");
              });
        // clang-format on
        os << '\n';
      }
    }

    os << "}\n";
  }
}
} // namespace kecc::ir
