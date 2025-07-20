#include "kecc/ir/Instruction.h"
#include "kecc/ir/Context.h"
#include "kecc/ir/IR.h"
#include "kecc/ir/IRAttributes.h"
#include "kecc/ir/IRInstructions.h"
#include "kecc/ir/JumpArg.h"
#include "kecc/ir/Value.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/MemAlloc.h"

namespace kecc::ir {

TypeID Instruction::getId() const {
  return getStorage()->getAbstractInstruction()->getId();
}

llvm::SMRange Instruction::getRange() const { return getStorage()->getRange(); }

void Instruction::print(IRPrintContext &context) const {
  getStorage()->print(context);
}

void Instruction::print(llvm::raw_ostream &os) const {
  IRPrintContext printContext(os);
  getParentBlock()->getParentFunction()->registerAllInstInfo(printContext);
  print(printContext);
}

void Instruction::dump() const { print(llvm::errs()); }

IRContext *Instruction::getContext() const {
  return getStorage()->getAbstractInstruction()->getContext();
}

Block *Instruction::getParentBlock() const {
  return getStorage()->getParentBlock();
}

void InstructionState::setOperands(llvm::ArrayRef<Value> operands) {
  this->operands.clear();
  this->operands.reserve(operands.size());
  for (const auto &op : operands) {
    this->operands.emplace_back(op);
  }
}

void InstructionState::pushOperand(Value operand) {
  this->operands.emplace_back(operand);
}

void InstructionState::setTypes(llvm::ArrayRef<Type> types) {
  this->types.clear();
  this->types.reserve(types.size());
  for (const auto &type : types) {
    pushType(type);
  }
}

void InstructionState::pushType(Type type) { pushType(type, {}); }
void InstructionState::pushType(Type type, llvm::StringRef name) {
  this->types.emplace_back(std::pair(type, name));
}

void InstructionState::setAttributes(llvm::ArrayRef<Attribute> attributes) {
  this->attributes.clear();
  this->attributes.reserve(attributes.size());
  for (const auto &attr : attributes) {
    this->attributes.emplace_back(attr);
  }
}

void InstructionState::pushAttribute(Attribute attribute) {
  this->attributes.emplace_back(attribute);
}

void InstructionState::setJumpArgs(llvm::ArrayRef<JumpArgState> jumpArgs) {
  this->jumpArgs.clear();
  this->jumpArgs.reserve(jumpArgs.size());
  for (const auto &arg : jumpArgs) {
    this->jumpArgs.emplace_back(arg);
  }
}

void InstructionState::pushJumpArg(const JumpArgState &jumpArg) {
  this->jumpArgs.emplace_back(jumpArg);
}

InstructionStorage::InstructionStorage(std::uint8_t resultSize,
                                       std::size_t operandSize,
                                       std::size_t attrSize,
                                       std::size_t jumpArgSize,
                                       llvm::SMRange range, Block *parentBlock)
    : resultSize(resultSize), operandSize(operandSize), attributeSize(attrSize),
      jumpArgSize(jumpArgSize), range(range), parentBlock(parentBlock),
      abstractInst(nullptr) {}

llvm::SMRange InstructionStorage::getRange() const { return range; }

InstructionStorage *InstructionStorage::create(InstructionState &state) {
  // Memory allocation and initialization logic for InstructionImpl
  // The Memory structure is like this:
  // Values | InstructionStorage | Operands | Attributes
  //        ^
  //      rawMem

  auto prefixBytes = llvm::alignTo(sizeof(ValueImpl) * state.types.size(),
                                   alignof(InstructionStorage));
  auto operandSize = state.operands.size();
  auto attrSize = state.attributes.size();
  auto jumpArgSize = state.jumpArgs.size();
  auto totalByteSize = totalSizeToAlloc<Operand, Attribute, JumpArg *>(
      operandSize, attrSize, jumpArgSize);

  auto sizeToMalloc = prefixBytes + totalByteSize;
  char *mallocMem = static_cast<char *>(llvm::safe_malloc(sizeToMalloc));
  char *rawMem = mallocMem + prefixBytes;

  InstructionStorage *impl = new (rawMem)
      InstructionStorage(state.types.size(), operandSize, attrSize, jumpArgSize,
                         state.range, state.parentBlock);

  Operand *operands = impl->getTrailingObjects<Operand>();
  for (std::size_t i = 0; i < operandSize; ++i) {
    new (&operands[i]) Operand(impl, state.operands[i]);
  }

  Attribute *attributes = impl->getTrailingObjects<Attribute>();
  std::uninitialized_copy(state.attributes.begin(), state.attributes.end(),
                          attributes);

  JumpArg **jumpArg = impl->getTrailingObjects<JumpArg *>();
  for (std::size_t i = 0; i < jumpArgSize; ++i) {
    jumpArg[i] = JumpArg::create(impl, state.jumpArgs[i]);
  }

  for (auto idx = 0u; idx < state.types.size(); ++idx) {
    auto [type, name] = state.types[idx];
    new (impl->getResultMemory(idx)) ValueImpl(name, idx, type);
  }

  return impl;
}

bool InstructionStorage::hasInterface(TypeID interfaceId) const {
  return abstractInst->hasInterface(interfaceId);
}

std::size_t InstructionStorage::getPrefixBytes() const {
  return llvm::alignTo(sizeof(ValueImpl) * resultSize,
                       alignof(InstructionStorage));
}

ValueImpl *InstructionStorage::getResultMemory(std::uint8_t valueNumber) {
  assert(valueNumber < resultSize && "Value number out of bounds");
  return reinterpret_cast<ValueImpl *>(this) - ++valueNumber;
}

void InstructionStorage::destroy() {
  // delete operand explicitly for update def-use chain
  for (std::size_t i = 0; i < operandSize; ++i)
    getTrailingObjects<Operand>()[i].~Operand();

  for (std::size_t i = 0; i < jumpArgSize; ++i)
    getTrailingObjects<JumpArg *>()[i]->destroy();

  for (std::size_t i = 0; i < resultSize; ++i)
    getResultMemory(i)->~ValueImpl();

  free(reinterpret_cast<char *>(this) - getPrefixBytes());
}

llvm::ArrayRef<Operand> InstructionStorage::getOperands() const {
  return llvm::ArrayRef<Operand>(getTrailingObjects<Operand>(), operandSize);
}

llvm::ArrayRef<Attribute> InstructionStorage::getAttributes() const {
  return {getTrailingObjects<Attribute>(), attributeSize};
}

Value InstructionStorage::getResult(std::uint8_t valueNumber) const {
  assert(valueNumber < resultSize && "Value number out of bounds");
  auto *memory =
      const_cast<InstructionStorage *>(this)->getResultMemory(valueNumber);
  return Value(memory);
}

llvm::SmallVector<Value> InstructionStorage::getResults() const {
  llvm::SmallVector<Value> results;
  results.reserve(resultSize);
  for (std::uint8_t i = 0; i < resultSize; ++i) {
    results.emplace_back(getResult(i));
  }
  return results;
}

const Operand &InstructionStorage::getOperand(std::size_t index) const {
  assert(index < operandSize && "Operand index out of bounds");
  return getTrailingObjects<Operand>()[index];
}

llvm::ArrayRef<JumpArg *> InstructionStorage::getJumpArgs() const {
  return llvm::ArrayRef<JumpArg *>(getTrailingObjects<JumpArg *>(),
                                   jumpArgSize);
}

Attribute InstructionStorage::getAttribute(std::size_t index) const {
  assert(index < attributeSize && "Attribute index out of bounds");
  return getTrailingObjects<Attribute>()[index];
}

JumpArg *InstructionStorage::getJumpArg(std::size_t index) const {
  assert(index < jumpArgSize && "JumpArg index out of bounds");
  return getTrailingObjects<JumpArg *>()[index];
}

void InstructionStorage::setAttribute(std::size_t index, Attribute attribute) {
  assert(index < attributeSize && "Attribute index out of bounds");
  getTrailingObjects<Attribute>()[index] = attribute;
}

void InstructionStorage::setOperand(std::size_t index, Value operand) {
  assert(index < operandSize && "Operand index out of bounds");
  getTrailingObjects<Operand>()[index] = Operand(this, operand);
}

void InstructionStorage::setJumpArg(std::size_t index, JumpArgState jumpArg) {
  assert(index < jumpArgSize && "JumpArg index out of bounds");
  JumpArg **jumpArgs = getTrailingObjects<JumpArg *>();
  jumpArgs[index]->destroy();
  jumpArgs[index] = JumpArg::create(this, jumpArg);
}

IRContext *InstructionStorage::getContext() const {
  return abstractInst->getContext();
}

WalkResult InstructionStorage::walk(
    llvm::function_ref<WalkResult(const Operand &)> callback) {
  Operand *operands = getTrailingObjects<Operand>();

  for (std::size_t i = 0; i < getOperands().size(); ++i) {
    auto result = callback(operands[i]);
    if (result.isInterrupt())
      return result;
    if (result.isSkip())
      return WalkResult::advance();
  }

  JumpArg **jumpArgs = getTrailingObjects<JumpArg *>();

  for (std::size_t i = 0; i < jumpArgSize; ++i) {
    JumpArg *jumpArg = jumpArgs[i];
    auto result = jumpArg->walk(callback);
    if (result.isInterrupt())
      return result;
    if (result.isSkip())
      return WalkResult::advance();
  }

  return WalkResult::advance();
}

void InstructionStorage::print(IRPrintContext &printContext) const {
  getAbstractInstruction()->getPrintFn()(this, printContext);
}

void InstructionStorage::replaceOperand(Value oldV, Value newV) {
  Operand *operands = getTrailingObjects<Operand>();

  for (std::size_t i = 0; i < operandSize; ++i) {
    if (operands[i] == oldV)
      operands[i] = Operand(this, newV);
  }

  for (std::size_t i = 0; i < jumpArgSize; ++i) {
    JumpArg *jumpArg = getTrailingObjects<JumpArg *>()[i];
    jumpArg->replaceOperand(oldV, newV);
  }
}

void InstructionStorage::dropReferences() {
  for (std::size_t i = 0; i < operandSize; ++i)
    getTrailingObjects<Operand>()[i].drop();

  for (std::size_t i = 0; i < jumpArgSize; ++i) {
    JumpArg *jumpArg = getTrailingObjects<JumpArg *>()[i];
    jumpArg->dropReferences();
  }
}

RegisterId IRPrintContext::getId(Value inst) const { return idMap.at(inst); }
void IRPrintContext::setId(Value inst, RegisterId id) {
  assert(idMap.find(inst) == idMap.end() && "ID already exists for this value");
  idMap.try_emplace(inst, id);
}

void IRPrintContext::printIndent() {
  os << '\n' << std::string(indent * indentWidth, ' ');
}

void IRPrintContext::clearIdMap() { idMap.clear(); }

//============================================================================//
/// Phi
//============================================================================//

void Phi::build(IRBuilder &builder, InstructionState &state, Type type) {
  state.pushType(type);
}

void Phi::printer(Phi op, IRPrintContext &context) {
  Value(op).printAsOperand(context, true);
}

//============================================================================//
/// Block Exit
//============================================================================//

bool BlockExit::classof(Instruction inst) {
  return inst.isa<inst::Jump, inst::Branch, inst::Switch, inst::Return,
                  inst::Unreachable>();
}

WalkResult BlockExit::walk(llvm::function_ref<WalkResult(JumpArg *)> callback) {

  JumpArg **jumpArgs = getStorage()->getTrailingObjects<JumpArg *>();
  for (std::size_t i = 0; i < getStorage()->getJumpArgSize(); ++i) {
    auto result = callback(jumpArgs[i]);
    if (result.isInterrupt())
      return result;
    if (result.isSkip())
      return WalkResult::advance();
  }
  return WalkResult::advance();
}

} // namespace kecc::ir
