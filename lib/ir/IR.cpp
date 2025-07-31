#include "kecc/ir/IR.h"
#include "kecc/ir/IRInstructions.h"
#include "kecc/ir/IRTypes.h"
#include "kecc/ir/Instruction.h"
#include "kecc/ir/WalkSupport.h"

namespace kecc::ir {

void IR::addFunction(Function *function) {
  assert(function && "Function cannot be null");
  Node *newNode = push(function);
  auto [_, inserted] = functionMap.try_emplace(function->getName(), newNode);
  assert(inserted && "Function with the same name already exists");
}

IR::~IR() {
  for (Function *function : *this) {
    function->dropReferences();
  }

  delete structBlock;
  delete globalBlock;
  delete constantBlock;
}

Function *IR::getFunction(llvm::StringRef name) const {
  auto it = functionMap.find(name);
  if (it != functionMap.end())
    return it->second->data;
  return nullptr;
}

void IR::erase(llvm::StringRef name) {
  Node *node = functionMap.lookup(name);
  assert(node && "Function not found in IR");
  node->remove();
  functionMap.erase(name);
}

void IR::print(IRPrintContext &printContext) const {
  bool first = true;
  for (const auto inst : *getStructBlock()) {
    if (first)
      first = false;
    else
      printContext.printIndent();

    inst->print(printContext);
  }

  for (const auto inst : *getGlobalBlock()) {
    if (first)
      first = false;
    else
      printContext.printIndent();
    inst->print(printContext);
  }

  if (printContext.isDebugMode()) {
    if (first)
      first = false;
    else
      printContext.printIndent();

    printContext.getOS() << "constants:";
    {
      IRPrintContext::AddIndent addIndent(printContext);
      for (const auto inst : *getConstantBlock()) {
        printContext.printIndent();
        inst->print(printContext);
      }
    }
  }

  for (auto I = begin(), E = end(); I != E; ++I) {
    auto *func = *I;
    if (first)
      first = false;
    else
      printContext.printIndent();
    printContext.printIndent();
    func->print(printContext);
  }
  if (!empty())
    printContext.printIndent();
}

Function::Function(llvm::StringRef name, Type functionType, IR *parentProgram,
                   IRContext *context)
    : name(name), functionType(functionType), parentProgram(parentProgram),
      context(context) {
  allocationBlock = new Block(-1);
  allocationBlock->setParentFunction(this);
  unresolvedBlock = new Block(-2);
  unresolvedBlock->setParentFunction(this);
}

Function::~Function() {
  dropReferences();

  delete allocationBlock;
  delete unresolvedBlock;
}

Block *Function::addBlock(int id) {
  auto it = blockMap.find(id);
  Block *block;
  if (it != blockMap.end()) {
    Node *node = it->second;
    block = node->data;
    node->data = nullptr;
    node->remove();
    blockMap.erase(it);
  } else {
    block = new Block(id);
    block->setParentFunction(this);
  }

  Node *newNode = push(block);
  blockMap[id] = newNode;
  return block;
}

Block *Function::getAllocationBlock() const { return allocationBlock; }

Block *Function::getEntryBlock() const {
  auto it = blockMap.find(entryBid);
  assert(it != blockMap.end() && "Entry block not found");
  return it->second->data;
}

WalkResult
Function::walk(llvm::function_ref<WalkResult(InstructionStorage *)> callback) {
  for (auto block : *this) {
    auto result = block->walk(callback);
    if (result.isInterrupt())
      return result;
    if (result.isSkip())
      break;
  }
  return WalkResult::advance();
}

WalkResult
Function::walk(llvm::function_ref<WalkResult(const Operand &)> callback) {
  auto result = walk([&](InstructionStorage *inst) -> WalkResult {
    return inst->walk(callback);
  });
  if (result.isInterrupt())
    return result;
  return WalkResult::advance();
}

void Function::print(IRPrintContext &printContext) const {
  printContext.clearIdMap();
  printContext.getOS() << "fun ";

  FunctionT funcT = functionType.cast<FunctionT>();
  for (auto I = funcT.getReturnTypes().begin(),
            E = funcT.getReturnTypes().end();
       I != E; ++I) {
    if (I != funcT.getReturnTypes().begin())
      printContext.getOS() << ", ";
    printContext.getOS() << (*I).toString();
  }

  printContext.getOS() << " @" << name << " (";
  for (auto I = funcT.getArgTypes().begin(), E = funcT.getArgTypes().end();
       I != E; ++I) {
    if (I != funcT.getArgTypes().begin())
      printContext.getOS() << ", ";
    printContext.getOS() << (*I).toString();
  }

  registerAllInstInfo(printContext);

  printContext.getOS() << ")";
  if (!hasDefinition())
    return;

  printContext.getOS() << " {";
  printContext.printIndent();
  printContext.getOS() << "init:";
  {
    IRPrintContext::AddIndent addIndent(printContext);
    printContext.printIndent();
    printContext.getOS() << "bid: b" << entryBid;
    printContext.printIndent();
    printContext.getOS() << "allocations:";
    {
      IRPrintContext::AddIndent addIndent(printContext);

      for (const auto inst : *getAllocationBlock()) {
        auto localVariable = inst->getDefiningInst<inst::LocalVariable>();
        printContext.printIndent();
        localVariable.printAsDef(printContext);
      }
    }
  }

  if (printContext.isDebugMode()) {
    printContext.printIndent();
    printContext.getOS() << "unresolved:";
    {
      IRPrintContext::AddIndent addIndent(printContext);
      for (const auto inst : *getUnresolvedBlock()) {
        auto unresolvedInst = inst->getDefiningInst<inst::Unresolved>();
        printContext.printIndent();
        unresolvedInst.print(printContext);
      }
    }
  }

  for (const auto block : *this) {
    printContext.printIndent();
    printContext.printIndent();
    block->print(printContext);
  }
  printContext.printIndent();
  printContext.getOS() << "}";
}

void Function::registerAllInstInfo(IRPrintContext &printContext) const {
  std::size_t allocCount = 0;
  for (const auto inst : *getAllocationBlock()) {
    auto localVariable = inst->getDefiningInst<inst::LocalVariable>();
    auto allocId = RegisterId::alloc(localVariable.getRange(), allocCount++);
    printContext.setId(localVariable.getResult(), allocId);
  }

  std::size_t unresolvedCount = 0;
  for (const auto inst : *getUnresolvedBlock()) {
    auto unresolvedInst = inst->getDefiningInst<inst::Unresolved>();
    auto unresolvedId =
        RegisterId::unresolved(unresolvedInst.getRange(), unresolvedCount++);
    printContext.setId(unresolvedInst.getResult(), unresolvedId);
  }

  for (const auto block : *this) {
    block->registerRid(printContext);
  }
}

Block *Function::getUnresolvedBlock() const { return unresolvedBlock; }

void Function::eraseBlock(int id) {
  auto it = blockMap.find(id);
  assert(it != blockMap.end() && "Block not found in function");
  Node *node = it->second;
  Block *block = node->data;
  node->remove();
  blockMap.erase(it);
}

void Function::dropReferences() {
  for (Block *block : *this) {
    block->walk([&](InstructionStorage *inst) -> WalkResult {
      inst->dropReferences();
      return WalkResult::advance();
    });
  }
}

bool Function::hasDefinition() const { return !empty(); }

} // namespace kecc::ir
