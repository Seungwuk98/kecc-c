#include "kecc/ir/Value.h"
#include "kecc/ir/IR.h"
#include "kecc/ir/IRInstructions.h"
#include "kecc/ir/Instruction.h"
#include "llvm/ADT/DenseSet.h"

namespace kecc::ir {

Operand::Operand(InstructionStorage *storage, Value value)
    : Value(value.getImpl()), owner(storage) {
  insertUsage(value.getImpl());
}

Operand::~Operand() { drop(); }

Operand &Operand::operator=(Operand &&other) noexcept {
  assert(other.impl && "Cannot assign from a null Operand");
  other.usageNode->data = this;
  impl = other.impl;
  usageNode->remove();
  usageNode = other.usageNode;
  other.usageNode = nullptr;
  other.impl = nullptr;
  return *this;
}

void Operand::insertUsage(ValueImpl *usageList) {
  assert(usageList && "Usage list cannot be null");
  auto *newNode = usageList->push(this);
  usageNode = newNode;
}

void Operand::drop() {
  if (!usageNode)
    return;
  usageNode->remove();
  usageNode = nullptr;
  impl = nullptr;
}

InstructionStorage *Value::getInstruction() const {
  std::uint8_t valueNumber = impl->getValueNumber();
  return reinterpret_cast<InstructionStorage *>(getImpl() + (++valueNumber));
}

void Value::replaceWith(Value newValue) {
  if (*this == newValue) {
    return; // No need to replace with itself
  }

  llvm::DenseSet<InstructionStorage *> userInsts;
  for (Operand *user : *getImpl()) {
    auto *owner = user->getOwner();
    if (userInsts.contains(owner))
      continue; // Skip if already processed

    userInsts.insert(owner);
  }

  for (InstructionStorage *inst : userInsts)
    inst->replaceOperand(*this, newValue);

  assert(getImpl()->empty() &&
         "All usages must be removed before replacing a value");
}

void Value::printAsOperand(IRPrintContext &context, bool printName) const {
  auto *storage = getInstruction();
  if (auto constant = storage->getDefiningInst<inst::Constant>()) {
    constant.print(context);
    return;
  }
  auto rid = context.getId(*this);
  context.getOS() << rid.toString() << ":" << getType();
  if (printName && !getValueName().empty())
    context.getOS() << ":" << getValueName();
}

void Value::printAsOperand(llvm::raw_ostream &os, bool printName) const {
  IRPrintContext context(os);
  if (auto constant = getDefiningInst<inst::Constant>()) {
    constant.print(context);
    return;
  }
  Block *parentBlock = getInstruction()->getParentBlock();
  Function *parentFunction = parentBlock->getParentFunction();
  parentFunction->registerAllInstInfo(context);
  printAsOperand(context);
}

} // namespace kecc::ir
