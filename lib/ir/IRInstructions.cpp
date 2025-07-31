#include "kecc/ir/IRInstructions.h"
#include "kecc/ir/Attribute.h"
#include "kecc/ir/IRAttributes.h"
#include "kecc/ir/IRTypes.h"
#include "kecc/ir/Instruction.h"
#include "kecc/ir/JumpArg.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/SourceMgr.h"
#include <format>
#include <string>

namespace kecc::ir::inst {

//============================================================================//
/// Nop
//============================================================================//

void Nop::build(IRBuilder &builder, InstructionState &state) {
  state.pushType(UnitT::get(builder.getContext()));
}

void Nop::printer(Nop op, IRPrintContext &context) {
  op.printAsOperand(context, true);
  context.getOS() << " = nop";
}

//============================================================================//
/// Load
//============================================================================//

void Load::build(IRBuilder &builder, InstructionState &state, Value ptr) {
  auto ptrType = ptr.getType().cast<PointerT>();
  state.pushType(ptrType.getPointeeType().constCanonicalize());
  state.pushOperand(ptr);
}

void Load::printer(Load op, IRPrintContext &context) {
  op.printAsOperand(context, true);
  context.getOS() << " = load ";
  op.getPointer().printAsOperand(context);
}

Value Load::getPointer() const { return this->getStorage()->getOperand(0); }
Value Load::Adaptor::getPointer() const { return operands[0]; }

//============================================================================//
/// Store
//============================================================================//

void Store::build(IRBuilder &builder, InstructionState &state, Value value,
                  Value ptr) {
  state.pushType(UnitT::get(builder.getContext()));
  state.pushOperand(value);
  state.pushOperand(ptr);
}

void Store::printer(Store op, IRPrintContext &context) {
  op.printAsOperand(context, true);
  context.getOS() << " = store ";

  op.getValue().printAsOperand(context);
  context.getOS() << ' ';
  op.getPointer().printAsOperand(context);
}

Value Store::getValue() const { return this->getStorage()->getOperand(0); }
Value Store::getPointer() const { return this->getStorage()->getOperand(1); }

Value Store::Adaptor::getValue() const { return operands[0]; }
Value Store::Adaptor::getPointer() const { return operands[1]; }

//============================================================================//
/// Call
//============================================================================//

void Call::build(IRBuilder &builder, InstructionState &state, Value function,
                 llvm::ArrayRef<Value> args) {
  auto funcType =
      function.getType().cast<PointerT>().getPointeeType().cast<FunctionT>();

  state.setTypes(funcType.getReturnTypes());
  state.pushOperand(function);
  for (const auto &arg : args) {
    state.pushOperand(arg);
  }
}

void Call::build(IRBuilder &builder, InstructionState &state, Value function,
                 llvm::ArrayRef<Value> args, llvm::ArrayRef<Type> resultTypes) {
  state.setTypes(resultTypes);
  state.pushOperand(function);
  for (const auto &arg : args) {
    state.pushOperand(arg);
  }
}

void Call::printer(Call op, IRPrintContext &context) {
  for (auto idx = 0u; idx < op.getNumResults(); ++idx) {
    if (idx > 0) {
      context.getOS() << ", ";
    }
    auto value = op.getStorage()->getResult(idx);
    value.printAsOperand(context, true);
  }

  context.getOS() << " = ";

  context.getOS() << "call ";
  op.getFunction().printAsOperand(context);
  context.getOS() << '(';
  for (auto I = op.getArguments().begin(), E = op.getArguments().end(); I != E;
       ++I) {
    if (I != op.getArguments().begin()) {
      context.getOS() << ", ";
    }
    I->printAsOperand(context);
  }
  context.getOS() << ')';
}

Value Call::getFunction() const { return this->getStorage()->getOperand(0); }
llvm::ArrayRef<Operand> Call::getArguments() const {
  return this->getStorage()->getOperands().slice(1);
}

Value Call::Adaptor::getFunction() const { return operands[0]; }
llvm::ArrayRef<Value> Call::Adaptor::getArguments() const {
  return operands.slice(1);
}

//============================================================================//
/// TypeCast
//============================================================================//

void TypeCast::build(IRBuilder &builder, InstructionState &state, Value value,
                     Type targetType) {
  state.pushType(targetType);
  state.pushOperand(value);
}

void TypeCast::printer(TypeCast op, IRPrintContext &context) {
  op.printAsOperand(context, true);
  context.getOS() << " = typecast ";

  op.getValue().printAsOperand(context);
  context.getOS() << " to " << op.getTargetType().toString();
}

Value TypeCast::getValue() const { return this->getStorage()->getOperand(0); }
Type TypeCast::getTargetType() const {
  return this->getStorage()->getResult(0).getType();
}

Value TypeCast::Adaptor::getValue() const { return operands[0]; }

//============================================================================//
/// Gep
//============================================================================//

void Gep::build(IRBuilder &builder, InstructionState &state, Value basePtr,
                Value offset, Type type) {
  state.pushType(type);
  state.pushOperand(basePtr);
  state.pushOperand(offset);
}

void Gep::printer(Gep op, IRPrintContext &context) {
  op.printAsOperand(context, true);
  context.getOS() << " = getelementptr ";
  op.getBasePointer().printAsOperand(context);
  context.getOS() << " offset ";
  op.getOffset().printAsOperand(context);
}

Value Gep::getBasePointer() const { return this->getStorage()->getOperand(0); }

Value Gep::getOffset() const { return this->getStorage()->getOperand(1); }

Value Gep::Adaptor::getBasePointer() const { return operands[0]; }
Value Gep::Adaptor::getOffset() const { return operands[1]; }

//============================================================================//
/// Binary
//==========================================================================//

void Binary::build(IRBuilder &builder, InstructionState &state, Value lhs,
                   Value rhs, OpKind opKind) {
  Type resultType;
  switch (opKind) {
  case Add:
  case Sub:
  case Mul:
  case Div:
  case Mod:
  case BitAnd:
  case BitOr:
  case BitXor:
  case Shl:
  case Shr:
    resultType = lhs.getType();
    break;
  case Eq:
  case Ne:
  case Lt:
  case Le:
  case Gt:
  case Ge:
    resultType = IntT::get(builder.getContext(), 1, true);
    break;
  }

  build(builder, state, lhs, rhs, opKind, resultType);
}

void Binary::build(IRBuilder &builder, InstructionState &state, Value lhs,
                   Value rhs, OpKind opKind, Type resultType) {
  state.pushType(resultType);
  state.pushOperand(lhs);
  state.pushOperand(rhs);
  state.pushAttribute(EnumAttr::get<OpKind>(builder.getContext(), opKind));
}

void Binary::printer(Binary op, IRPrintContext &context) {
  op.printAsOperand(context, true);
  context.getOS() << " = ";

  switch (op.getOpKind()) {
#define CASE(opKind, opName)                                                   \
  case OpKind::opKind:                                                         \
    context.getOS() << #opName;                                                \
    break;
    CASE(Add, add);
    CASE(Sub, sub);
    CASE(Mul, mul);
    CASE(Div, div);
    CASE(Mod, mod);
    CASE(Shl, shl);
    CASE(Shr, shr);
    CASE(BitAnd, and);
    CASE(BitOr, or);
    CASE(BitXor, xor);
#undef CASE
#define CASE(opKind, opName)                                                   \
  case OpKind::opKind:                                                         \
    context.getOS() << "cmp " << #opName;                                      \
    break;
    CASE(Eq, eq);
    CASE(Ne, ne);
    CASE(Lt, lt);
    CASE(Le, le);
    CASE(Gt, gt);
    CASE(Ge, ge);
#undef CASE
  }

  context.getOS() << ' ';
  op.getLhs().printAsOperand(context);
  context.getOS() << ' ';
  op.getRhs().printAsOperand(context);
}

Value Binary::getLhs() const { return this->getStorage()->getOperand(0); }
Value Binary::getRhs() const { return this->getStorage()->getOperand(1); }
Binary::OpKind Binary::getOpKind() const {
  return this->getStorage()
      ->getAttribute(0)
      .cast<EnumAttr>()
      .getEnumValue<OpKind>();
}

Value Binary::Adaptor::getLhs() const { return operands[0]; }
Value Binary::Adaptor::getRhs() const { return operands[1]; }

//============================================================================//
/// Unary
//============================================================================//

void Unary::build(IRBuilder &builder, InstructionState &state, Value value,
                  OpKind op) {
  Type resultType = value.getType();
  state.pushType(resultType);
  state.pushOperand(value);
  state.pushAttribute(EnumAttr::get<OpKind>(builder.getContext(), op));
}

void Unary::printer(Unary op, IRPrintContext &context) {
  op.printAsOperand(context, true);
  context.getOS() << " = ";

  switch (op.getOpKind()) {
  case OpKind::Plus:
    context.getOS() << "plus ";
    break;
  case OpKind::Minus:
    context.getOS() << "minus ";
    break;
  case OpKind::Negate:
    context.getOS() << "negate ";
    break;
  }

  op.getValue().printAsOperand(context);
}

Value Unary::getValue() const { return this->getStorage()->getOperand(0); }
Unary::OpKind Unary::getOpKind() const {
  return this->getStorage()
      ->getAttribute(0)
      .cast<EnumAttr>()
      .getEnumValue<OpKind>();
}

Value Unary::Adaptor::getValue() const { return operands[0]; }

//============================================================================//
/// Jump
//============================================================================//

void Jump::build(IRBuilder &builder, InstructionState &state,
                 JumpArgState arg) {
  state.pushJumpArg(arg);
}

void Jump::printer(Jump op, IRPrintContext &context) {
  context.getOS() << "j ";
  op.getJumpArg()->printJumpArg(context);
}

JumpArg *Jump::getJumpArg() const { return this->getStorage()->getJumpArg(0); }

const JumpArgState &Jump::Adaptor::getJumpArg() const { return jumpArgs[0]; }

//============================================================================//
/// Branch
//============================================================================//

void Branch::build(IRBuilder &builder, InstructionState &state, Value condition,
                   JumpArgState ifArg, JumpArgState elseArg) {
  state.pushOperand(condition);
  state.pushJumpArg(ifArg);
  state.pushJumpArg(elseArg);
}

void Branch::printer(Branch op, IRPrintContext &context) {
  context.getOS() << "br ";

  op.getCondition().printAsOperand(context);
  context.getOS() << ", ";
  op.getIfArg()->printJumpArg(context);
  context.getOS() << ", ";
  op.getElseArg()->printJumpArg(context);
}

Value Branch::getCondition() const { return this->getStorage()->getOperand(0); }
JumpArg *Branch::getIfArg() const { return this->getStorage()->getJumpArg(0); }
JumpArg *Branch::getElseArg() const {
  return this->getStorage()->getJumpArg(1);
}

Value Branch::Adaptor::getCondition() const { return operands[0]; }
JumpArgState Branch::Adaptor::getIfArg() const { return jumpArgs[0]; }
JumpArgState Branch::Adaptor::getElseArg() const { return jumpArgs[1]; }

//============================================================================//
/// Switch
//============================================================================//

void Switch::build(IRBuilder &builder, InstructionState &state, Value value,
                   llvm::ArrayRef<Value> caseValues,
                   llvm::ArrayRef<JumpArgState> cases,
                   JumpArgState defaultCase) {
  state.pushOperand(value);
  for (const auto &caseValue : caseValues) {
    state.pushOperand(caseValue);
  }

  state.pushJumpArg(defaultCase);
  for (const auto &caseArg : cases) {
    state.pushJumpArg(caseArg);
  }
}

void Switch::printer(Switch op, IRPrintContext &context) {
  context.getOS() << "switch ";
  op.getValue().printAsOperand(context);
  context.getOS() << " default ";
  op.getDefaultCase()->printJumpArg(context);
  context.getOS() << " [";

  {
    IRPrintContext::AddIndent addIndent(context);

    for (auto idx = 0u; idx < op.getCaseSize(); ++idx) {
      Value caseValue = op.getCaseValue(idx);
      JumpArg *caseArg = op.getCaseJumpArg(idx);
      context.printIndent();
      caseValue.printAsOperand(context);
      context.getOS() << ' ';
      caseArg->printJumpArg(context);
    }
  }
  context.printIndent();
  context.getOS() << ']';
}

Value Switch::getValue() const { return this->getStorage()->getOperand(0); }
Value Switch::getCaseValue(std::size_t idx) const {
  return this->getStorage()->getOperands().slice(1)[idx];
}
JumpArg *Switch::getCaseJumpArg(std::size_t idx) const {
  return this->getStorage()->getJumpArgs().slice(1)[idx];
}
JumpArg *Switch::getDefaultCase() const {
  return this->getStorage()->getJumpArg(0);
}
std::size_t Switch::getCaseSize() const {
  assert(this->getStorage()->getOperands().size() - 1 ==
             this->getStorage()->getJumpArgSize() - 1 &&
         "Number of case values and jump args must match");
  return this->getStorage()->getJumpArgSize() - 1;
}

Value Switch::Adaptor::getValue() const { return operands[0]; }
Value Switch::Adaptor::getCaseValue(std::size_t idx) const {
  return operands[idx + 1];
}
JumpArgState Switch::Adaptor::getCaseJumpArg(std::size_t idx) const {
  return jumpArgs[idx + 1];
}
JumpArgState Switch::Adaptor::getDefaultCase() const { return jumpArgs[0]; }
std::size_t Switch::Adaptor::getCaseSize() const { return jumpArgs.size() - 1; }

//============================================================================//
/// Return
//============================================================================//

void Return::build(IRBuilder &builder, InstructionState &state) {}
void Return::build(IRBuilder &builder, InstructionState &state, Value value) {
  state.pushOperand(value);
}
void Return::build(IRBuilder &builder, InstructionState &state,
                   llvm::ArrayRef<Value> value) {
  state.setOperands(value);
}

void Return::printer(Return op, IRPrintContext &context) {
  context.getOS() << "ret";

  if (op.getValueSize() > 0) {
    context.getOS() << ' ';
    for (auto idx = 0u; idx < op.getValueSize(); ++idx) {
      if (idx != 0u)
        context.getOS() << ", ";
      auto value = op.getValue(idx);
      value.printAsOperand(context);
    }
  }
}

Value Return::getValue(std::size_t idx) const {
  assert(idx < getValueSize() && "Index out of bounds for return value");
  return getValues()[idx];
}

llvm::ArrayRef<Operand> Return::getValues() const {
  return this->getStorage()->getOperands();
}

Value Return::Adaptor::getValue(std::size_t idx) const {
  assert(idx < getValueSize() && "Index out of bounds for return value");
  return operands[idx];
}
llvm::ArrayRef<Value> Return::Adaptor::getValues() const { return operands; }

//============================================================================//
/// Unreachable
//============================================================================//

void Unreachable::build(IRBuilder &builder, InstructionState &state) {}

void Unreachable::printer(Unreachable op, IRPrintContext &context) {
  context.getOS() << "unreachable";
}

//============================================================================//
/// Constant
//============================================================================//

static void printConstValue(ConstantAttr value, IRPrintContext &context) {
  llvm::TypeSwitch<ConstantAttr>(value)
      .Case([&](ConstantIntAttr value) {
        if (value.isSigned())
          context.getOS() << static_cast<std::int64_t>(value.getValue());
        else
          context.getOS() << value.getValue();
      })
      .Case([&](ConstantFloatAttr value) {
        llvm::SmallVector<char, 20> buffer;
        value.getValue().toString(buffer, 17, 17);
        context.getOS() << buffer;
      })
      .Case([&](ConstantStringFloatAttr value) {
        context.getOS() << value.getValue();
      })
      .Case([&](ConstantUndefAttr value) { context.getOS() << "undef"; })
      .Case([&](ConstantUnitAttr value) { context.getOS() << "unit"; })
      .Case([&](ConstantVariableAttr value) {
        context.getOS() << "@" << value.getName();
      })
      .Default(
          [&](ConstantAttr) { llvm_unreachable("Unsupported constant type"); });
}

void Constant::build(IRBuilder &builder, InstructionState &state,
                     ConstantAttr value) {
  state.pushType(value.getType());
  state.pushAttribute(value);
}

void Constant::printer(Constant op, IRPrintContext &context) {
  printConstValue(op.getValue(), context);
  context.getOS() << ':' << op.getValue().getType();
}

ConstantAttr Constant::getValue() const {
  return this->getStorage()->getAttribute(0).cast<ConstantAttr>();
}

void Constant::replaceValue(ConstantAttr value) {
  assert(getType() == value.getType() &&
         "Type mismatch in constant replacement");
  this->getStorage()->setAttribute(0, value);
}

//============================================================================//
/// StructDefinition
//============================================================================//

void StructDefinition::build(
    IRBuilder &builder, InstructionState &state,
    llvm::ArrayRef<std::pair<llvm::StringRef, Type>> fields,
    llvm::StringRef name) {
  state.pushAttribute(StringAttr::get(builder.getContext(), name));
  for (const auto &field : fields) {
    state.pushAttribute(StringAttr::get(builder.getContext(), field.first));
    state.pushAttribute(TypeAttr::get(builder.getContext(), field.second));
  }
}

void StructDefinition::printer(StructDefinition op, IRPrintContext &context) {
  context.getOS() << "struct " << op.getName() << " : ";

  if (!op.getFieldSize()) {
    context.getOS() << "opaque";
  } else {
    context.getOS() << "{ ";
    {
      IRPrintContext::AddIndent addIndent(context);
      for (auto idx = 0u; idx < op.getFieldSize(); ++idx) {
        if (idx > 0) {
          context.getOS() << ", ";
        }
        auto field = op.getField(idx);
        context.getOS() << (field.first.empty() ? "%anon" : field.first) << ":"
                        << field.second.toString();
      }
    }
    context.getOS() << " }";
  }
}

std::pair<llvm::StringRef, Type> StructDefinition::getField(size_t idx) const {
  assert(idx < this->getFieldSize() && "Index out of bounds");
  return {this->getStorage()
              ->getAttribute((idx << 1) + 1)
              .cast<StringAttr>()
              .getValue(),
          this->getStorage()
              ->getAttribute((idx + 1) << 1)
              .cast<TypeAttr>()
              .getType()};
}

size_t StructDefinition::getFieldSize() const {
  return (this->getStorage()->getAttributes().size() - 1) >> 1;
}

llvm::StringRef StructDefinition::getName() const {
  return this->getStorage()->getAttribute(0).cast<StringAttr>().getValue();
}

//============================================================================//
/// GlobalVariableDefinition
//============================================================================//

void GlobalVariableDefinition::build(IRBuilder &builder,
                                     InstructionState &state, Type type,
                                     llvm::StringRef name) {
  state.pushAttribute(StringAttr::get(builder.getContext(), name));
  state.pushAttribute(TypeAttr::get(builder.getContext(), type));
}

void GlobalVariableDefinition::build(IRBuilder &builder,
                                     InstructionState &state, Type type,
                                     llvm::StringRef name,
                                     InitializerAttr initializer) {
  state.pushAttribute(StringAttr::get(builder.getContext(), name));
  state.pushAttribute(TypeAttr::get(builder.getContext(), type));
  state.pushAttribute(initializer);
}

static void printInterpretedInitailier(Attribute attr,
                                       IRPrintContext &context) {
  llvm::TypeSwitch<Attribute>(attr)
      .Case([&](ConstantAttr value) { printConstValue(value, context); })
      .Case([&](ArrayAttr array) {
        context.getOS() << "{";
        for (auto idx = 0u; idx < array.getValues().size(); ++idx) {
          if (idx > 0) {
            context.getOS() << ", ";
          }
          auto value = array.getValues()[idx];
          printInterpretedInitailier(value, context);
        }
        context.getOS() << "}";
      })
      .Default([&](Attribute) {
        llvm_unreachable("Unsupported initializer type for global variable");
      });
}

void GlobalVariableDefinition::printer(GlobalVariableDefinition op,
                                       IRPrintContext &context) {
  context.getOS() << "var " << op.getType() << " @" << op.getName();

  if (op.getInitializer()) {
    auto init = op.getInitializer();
    context.getOS() << " = ";
    if (auto ast = init.dyn_cast<InitializerAttr>())
      ast.printInitializer(context);
    else {
      printInterpretedInitailier(init, context);
    }
  }
}

llvm::StringRef GlobalVariableDefinition::getName() const {
  return this->getStorage()->getAttribute(0).cast<StringAttr>().getValue();
}

Type GlobalVariableDefinition::getType() const {
  return this->getStorage()->getAttribute(1).cast<TypeAttr>().getType();
}

Attribute GlobalVariableDefinition::getInitializer() const {
  if (this->getStorage()->getAttributes().size() < 3) {
    return nullptr;
  }
  return this->getStorage()->getAttribute(2).cast<InitializerAttr>();
}

void GlobalVariableDefinition::interpretInitializer() {
  if (auto initializer = this->getInitializer()) {
    if (auto astInitializer = initializer.dyn_cast<ASTInitializerList>()) {
      auto interpreted = astInitializer.interpret();
      if (!interpreted) {
        getContext()->diag().report(
            getRange(), llvm::SourceMgr::DK_Error,
            "Interpret failed for initializer of global variable @" +
                this->getName().str());
      }

      getStorage()->setAttribute(2, interpreted);
    }
  }
}

//============================================================================//
/// LocalVariable
//============================================================================//

void LocalVariable::build(IRBuilder &builder, InstructionState &state,
                          Type type) {
  state.pushType(type);
}

void LocalVariable::printAsDef(IRPrintContext &context) const {
  auto rid = context.getId(*this);
  context.getOS() << rid.toString() << ":"
                  << getType().cast<PointerT>().getPointeeType();

  auto valueName = Value(*this).getValueName();
  if (!valueName.empty()) {
    context.getOS() << ":" << valueName;
  }
}

void LocalVariable::printer(LocalVariable op, IRPrintContext &context) {
  Value(op).printAsOperand(context, true);
}

//============================================================================//
/// Unresolved
//============================================================================//

void Unresolved::build(IRBuilder &builder, InstructionState &state, Type type) {
  state.pushType(type);
}

void Unresolved::printer(Unresolved op, IRPrintContext &context) {
  auto rid = context.getId(op);
  context.getOS() << rid.toString() << ":" << op.getType() << " = unresolved";
}

//============================================================================//
/// Outline constant
//============================================================================//

void OutlineConstant::build(IRBuilder &builder, InstructionState &state,
                            Value value) {
  assert(value.getDefiningInst<Constant>() && "Requiring constant operand");
  state.pushType(value.getType());
  state.pushOperand(value);
}

Value OutlineConstant::getConstant() const {
  return getStorage()->getOperand(0);
}

void OutlineConstant::printer(OutlineConstant op, IRPrintContext &context) {
  op.printAsOperand(context, true);
  context.getOS() << " = outline ";
  op.getConstant().printAsOperand(context);
}

Value OutlineConstant::Adaptor::getConstant() const { return operands[0]; }

//============================================================================//
/// inline call
//============================================================================//

void InlineCall::build(IRBuilder &builder, InstructionState &state,
                       llvm::StringRef name, FunctionT type,
                       llvm::ArrayRef<Value> args) {
  auto retTypes = type.getReturnTypes();
  state.setTypes(retTypes);

  state.pushAttribute(StringAttr::get(builder.getContext(), name));
  state.pushAttribute(TypeAttr::get(builder.getContext(), type));

  state.setOperands(args);
}

llvm::StringRef InlineCall::getName() const {
  return this->getStorage()->getAttribute(0).cast<StringAttr>().getValue();
}

FunctionT InlineCall::getFunctionType() const {
  return this->getStorage()
      ->getAttribute(1)
      .cast<TypeAttr>()
      .getType()
      .cast<FunctionT>();
}
llvm::ArrayRef<Operand> InlineCall::getArguments() const {
  return this->getStorage()->getOperands();
}

void InlineCall::printer(InlineCall op, IRPrintContext &context) {
  for (auto idx = 0u; idx < op.getNumResults(); ++idx) {
    if (idx > 0)
      context.getOS() << ", ";
    auto value = op.getStorage()->getResult(idx);
    value.printAsOperand(context, true);
  }

  context.getOS() << " = inline call @" << op.getName() << ":"
                  << op.getFunctionType() << "(";

  for (auto I = op.getArguments().begin(), E = op.getArguments().end(); I != E;
       ++I) {
    if (I != op.getArguments().begin())
      context.getOS() << ", ";
    I->printAsOperand(context);
  }
  context.getOS() << ')';
}

void registerBuiltinInstructions(IRContext *context) {
  context->registerInst<Phi>();
  context->registerInst<inst::Nop>();
  context->registerInst<inst::Load>();
  context->registerInst<inst::Store>();
  context->registerInst<inst::Call>();
  context->registerInst<inst::TypeCast>();
  context->registerInst<inst::Gep>();
  context->registerInst<inst::Binary>();
  context->registerInst<inst::Unary>();
  context->registerInst<inst::Jump>();
  context->registerInst<inst::Branch>();
  context->registerInst<inst::Switch>();
  context->registerInst<inst::Return>();
  context->registerInst<inst::Unreachable>();
  context->registerInst<inst::Constant>();
  context->registerInst<inst::StructDefinition>();
  context->registerInst<inst::GlobalVariableDefinition>();
  context->registerInst<inst::LocalVariable>();
  context->registerInst<inst::Unresolved>();
  context->registerInst<inst::InlineCall>();
}

} // namespace kecc::ir::inst
