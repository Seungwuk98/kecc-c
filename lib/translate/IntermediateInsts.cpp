#include "kecc/translate/IntermediateInsts.h"
#include "kecc/ir/IRAttributes.h"
#include "kecc/ir/Type.h"
#include "kecc/ir/Value.h"

DEFINE_KECC_TYPE_ID(kecc::translate::inst::LoadOffset)
DEFINE_KECC_TYPE_ID(kecc::translate::inst::StoreOffset)
DEFINE_KECC_TYPE_ID(kecc::translate::inst::Copy)

namespace kecc {
namespace translate::inst {

void LoadOffset::build(ir::IRBuilder &builder, ir::InstructionState &state,
                       ir::Value ptr, std::int64_t offset, ir::Type loadType) {
  state.pushType(loadType);
  state.pushOperand(ptr);
  state.pushAttribute(
      ir::ConstantIntAttr::get(builder.getContext(), offset, 64, true));
}

ir::Value LoadOffset::getPointer() const { return getPointerAsOperand(); }
const ir::Operand &LoadOffset::getPointerAsOperand() const {
  return getStorage()->getOperand(0);
}
std::int64_t LoadOffset::getOffset() const {
  auto intAttr = getStorage()->getAttribute(0).cast<ir::ConstantIntAttr>();
  return intAttr.getValue();
}

void LoadOffset::printer(LoadOffset op, ir::IRPrintContext &context) {
  context.printValue(op);
  context.getOS() << " = load ";
  context.printOperand(op.getPointerAsOperand());
  context.getOS() << " offset " << op.getOffset();
}

void StoreOffset::build(ir::IRBuilder &builder, ir::InstructionState &state,
                        ir::Value value, ir::Value base, std::int64_t offset) {
  state.pushOperand(value);
  state.pushOperand(base);
  state.pushAttribute(
      ir::ConstantIntAttr::get(builder.getContext(), offset, 64, true));
}

const ir::Operand &StoreOffset::getValueAsOperand() const {
  return getStorage()->getOperand(0);
}
const ir::Operand &StoreOffset::getPointerAsOperand() const {
  return getStorage()->getOperand(1);
}
std::int64_t StoreOffset::getOffset() const {
  auto intAttr = getStorage()->getAttribute(0).cast<ir::ConstantIntAttr>();
  return intAttr.getValue();
}

void StoreOffset::printer(StoreOffset op, ir::IRPrintContext &context) {
  context.getOS() << "store ";
  context.printOperand(op.getValueAsOperand());
  context.getOS() << ' ';
  context.printOperand(op.getPointerAsOperand());
  context.getOS() << " offset " << op.getOffset();
}

//============================================================================//
/// Copy
//============================================================================//

void Copy::build(ir::IRBuilder &builder, ir::InstructionState &state,
                 ir::Value value, ir::Type type) {
  state.pushType(type);
  state.pushOperand(value);
}

void Copy::printer(Copy op, ir::IRPrintContext &context) {
  context.printValue(op, true);
  context.getOS() << " = copy ";
  context.printOperand(op.getValueAsOperand());
}

ir::Value Copy::getValue() const { return getValueAsOperand(); }
const ir::Operand &Copy::getValueAsOperand() const {
  return this->getStorage()->getOperand(0);
}

ir::Value Copy::Adaptor::getValue() const { return operands[0]; }

} // namespace translate::inst

} // namespace kecc
