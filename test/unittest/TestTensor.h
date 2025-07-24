#include "kecc/ir/IRAttributes.h"
#include "kecc/ir/IRBuilder.h"
#include "kecc/ir/IRInstructions.h"
#include "kecc/ir/IRTypes.h"
#include "kecc/ir/Instruction.h"
#include "kecc/ir/PatternMatch.h"
#include "kecc/ir/Type.h"
#include "llvm/ADT/SmallVectorExtras.h"

namespace kecc::ir {

class TensorTImpl
    : public TypeImplTemplate<std::pair<Type, llvm::ArrayRef<size_t>>> {
public:
  static TensorTImpl *create(TypeStorage *storage, const KeyTy &key) {
    return create(storage, key.first, key.second);
  }

  static TensorTImpl *create(TypeStorage *storage, Type elementType,
                             llvm::ArrayRef<size_t> shape) {
    auto copiedShape = storage->copyArray(shape);
    auto *impl = new (storage->allocate<TensorTImpl>())
        TensorTImpl(elementType, copiedShape);
    return impl;
  }

  Type getElementType() const { return getKeyValue().first; }
  llvm::ArrayRef<size_t> getShape() const { return getKeyValue().second; }

private:
  TensorTImpl(Type elementType, llvm::ArrayRef<size_t> shape)
      : TypeImplTemplate({elementType, shape}) {}
};

class TensorT : public Type::Base<TensorT, Type, TensorTImpl> {
public:
  using Base::Base;

  static TensorT get(IRContext *context, Type elementType,
                     llvm::ArrayRef<size_t> shape) {
    return Base::get(context, elementType, shape);
  }

  Type getElementType() const { return getImpl()->getElementType(); }
  llvm::ArrayRef<size_t> getShape() const { return getImpl()->getShape(); }

  static void printer(TensorT type, llvm::raw_ostream &os) {
    os << "tensor<" << type.getElementType() << ", (";
    auto shape = type.getShape();
    for (std::size_t i = 0; i < shape.size(); ++i) {
      os << shape[i];
      if (i < shape.size() - 1) {
        os << ", ";
      }
    }
    os << ")>";
  }
};

namespace inst {

class CreateTensor
    : public InstructionTemplate<CreateTensor, Instruction, OneResult> {
public:
  static void build(IRBuilder &builder, InstructionState &state,
                    llvm::ArrayRef<size_t> shape,
                    llvm::ArrayRef<Value> operands) {
    assert(llvm::all_equal(llvm::map_to_vector(
               operands, [](const Value &v) { return v.getType(); })) &&
           "All operands must have the same type");
    assert(!shape.empty() && "Shape must not be empty");

    size_t elementCount = 1;
    for (size_t dim : shape)
      elementCount *= dim;

    assert(operands.size() == elementCount &&
           "Number of operands must match the total number of elements in the "
           "tensor shape");

    Type elementType = operands[0].getType();
    TensorT tensorT = TensorT::get(builder.getContext(), elementType, shape);

    state.pushType(tensorT);
    state.setOperands(operands);
  }

  llvm::ArrayRef<size_t> getShape() const {
    return getType().cast<TensorT>().getShape();
  }
  Type getElementType() const {
    return getType().cast<TensorT>().getElementType();
  }
  llvm::ArrayRef<Operand> getOperands() const {
    return getStorage()->getOperands();
  }

  static void printer(CreateTensor inst, IRPrintContext &context) {
    inst.printAsOperand(context, true);
    context.getOS() << " = tensor [";
    for (size_t i = 0; i < inst->getOperands().size(); ++i) {
      if (i > 0)
        context.getOS() << ", ";
      inst->getOperands()[i].printAsOperand(context.getOS());
    }
    context.getOS() << "]";
  }

  struct Adaptor {
    Adaptor(llvm::ArrayRef<Value> operands, llvm::ArrayRef<JumpArgState>)
        : operands(operands) {}

    llvm::ArrayRef<Value> getOperands() const { return operands; }

    llvm::ArrayRef<Value> operands;
  };
};

class Transpose
    : public InstructionTemplate<Transpose, Instruction, OneResult> {
public:
  static void build(IRBuilder &builder, InstructionState &state, Value input) {
    state.pushOperand(input);

    TensorT inputTensorT = input.getType().cast<TensorT>();
    llvm::ArrayRef<size_t> inputShape = inputTensorT.getShape();
    llvm::SmallVector<size_t> outputShape(inputShape.rbegin(),
                                          inputShape.rend());
    TensorT outputTensorT = TensorT::get(
        builder.getContext(), inputTensorT.getElementType(), outputShape);
    state.pushType(outputTensorT);
  }

  llvm::ArrayRef<size_t> getShape() const {
    return getType().cast<TensorT>().getShape();
  }
  Type getElementType() const {
    return getType().cast<TensorT>().getElementType();
  }
  Value getInput() const { return getStorage()->getOperands()[0]; }
  static void printer(Transpose inst, IRPrintContext &context) {
    inst.printAsOperand(context, true);
    context.getOS() << " = transpose ";
    inst.getInput().printAsOperand(context.getOS());
  }

  struct Adaptor {
    Adaptor(llvm::ArrayRef<Operand> operands, llvm::ArrayRef<JumpArgState>)
        : operands(operands) {}

    Value getInput() const { return operands[0]; }

    llvm::ArrayRef<Operand> operands;
  };
};
} // namespace inst

void addTransposePattern(PatternSet &pattern);

} // namespace kecc::ir

DECLARE_KECC_TYPE_ID(kecc::ir::TensorT);
DECLARE_KECC_TYPE_ID(kecc::ir::inst::CreateTensor);
DECLARE_KECC_TYPE_ID(kecc::ir::inst::Transpose);
