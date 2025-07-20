#include "TestUtils.h"
#include "kecc/ir/IRAttributes.h"
#include "kecc/ir/IRBuilder.h"
#include "kecc/ir/IRInstructions.h"
#include "kecc/ir/IRTypes.h"
#include "kecc/ir/Module.h"
#include "kecc/ir/Type.h"

namespace kecc::ir {

TEST_CASE("Module Tests") {
  IRContext context;

  SUBCASE("Module") {
    auto ir = std::make_unique<IR>(&context);

    auto unitT = UnitT::get(&context);
    auto i32 = IntT::get(&context, 32, true);
    auto funcType = FunctionT::get(&context, unitT, {});

    Function *func = new Function("test_func", funcType, ir.get(), &context);

    ir->addFunction(func);

    Block *block = func->addBlock(0);
    func->setEntryBlock(0);

    IRBuilder builder(&context);
    builder.setInsertionPoint(ir->getConstantBlock());

    Value const15 = builder.create<inst::Constant>(
        {}, ConstantIntAttr::get(&context, 15, 32, true));

    Value const20 = builder.create<inst::Constant>(
        {}, ConstantIntAttr::get(&context, 20, 32, true));

    builder.setInsertionPoint(block);

    auto add =
        builder.create<inst::Binary>({}, const15, const20, inst::Binary::Add);

    auto ret = builder.create<inst::Return>({}, llvm::ArrayRef<Value>(add));

    std::string result;
    llvm::raw_string_ostream ss(result);
    IRPrintContext printContext(ss, IRPrintContext::Debug);

    ir->print(printContext);
    STR_EQ(result, R"(
constants:
  15:i32
  20:i32

fun unit @test_func () {
init:
  bid: b0
  allocations:
unresolved:

block b0:
  %b0:i0:i32 = add 15:i32 20:i32
  ret %b0:i0:i32
})");

    auto module = Module::create(std::move(ir));

    // replace 15 to 30
    builder.setInsertionPoint(module->getIR()->getConstantBlock());
    auto const30 = builder.create<inst::Constant>(
        {}, ConstantIntAttr::get(&context, 30, 32, true));

    const15.replaceWith(const30);

    result.clear();
    module->getIR()->print(printContext);

    STR_EQ(result, R"(
constants:
  15:i32
  20:i32
  30:i32

fun unit @test_func () {
init:
  bid: b0
  allocations:
unresolved:

block b0:
  %b0:i0:i32 = add 30:i32 20:i32
  ret %b0:i0:i32
})");

    // replace 20 to 15
    const20.replaceWith(const15);

    result.clear();
    module->getIR()->print(printContext);
    STR_EQ(result, R"(
constants:
  15:i32
  20:i32
  30:i32

fun unit @test_func () {
init:
  bid: b0
  allocations:
unresolved:

block b0:
  %b0:i0:i32 = add 30:i32 15:i32
  ret %b0:i0:i32
})");
  }

  SUBCASE("Unresolved Instructions") {
    auto ir = std::make_unique<IR>(&context);

    auto unitT = UnitT::get(&context);
    auto i32 = IntT::get(&context, 32, true);

    auto calleeFuncType = FunctionT::get(&context, {i32, i32}, {});
    Function *calleeFunc =
        new Function("callee_func", calleeFuncType, ir.get(), &context);

    auto callerFuncType = FunctionT::get(&context, unitT, {});
    Function *callerFunc =
        new Function("caller_func", callerFuncType, ir.get(), &context);

    ir->addFunction(calleeFunc);
    ir->addFunction(callerFunc);
    Block *calleeBlock = calleeFunc->addBlock(0);
    calleeFunc->setEntryBlock(0);

    Block *callerBlock = callerFunc->addBlock(0);
    callerFunc->setEntryBlock(0);

    IRBuilder builder(&context);
    builder.setInsertionPoint(ir->getConstantBlock());

    auto const15 = builder.create<inst::Constant>(
        {}, ConstantIntAttr::get(&context, 15, 32, true));
    auto const20 = builder.create<inst::Constant>(
        {}, ConstantIntAttr::get(&context, 20, 32, true));
    auto calleeAttr = ConstantVariableAttr::get(
        &context, "callee_func", PointerT::get(&context, calleeFuncType));
    auto calleeConst = builder.create<inst::Constant>({}, calleeAttr);

    builder.setInsertionPoint(calleeBlock);
    auto ret = builder.create<inst::Return>(
        {}, llvm::ArrayRef<Value>{const15, const20});

    builder.setInsertionPoint(callerFunc->getUnresolvedBlock());
    auto u0 = builder.create<inst::Unresolved>({}, i32);
    auto u1 = builder.create<inst::Unresolved>({}, i32);

    builder.setInsertionPoint(callerBlock);

    auto call = builder.create<inst::Call>({}, calleeConst, std::nullopt);
    auto add = builder.create<inst::Binary>({}, u0, u1, inst::Binary::Add);

    auto retCall = builder.create<inst::Return>({}, add);

    std::string result;
    llvm::raw_string_ostream ss(result);
    IRPrintContext printContext(ss, IRPrintContext::Debug);
    ir->print(printContext);

    STR_EQ(result, R"(
constants:
  15:i32
  20:i32
  @callee_func:[ret:i32, i32 params:()]*

fun i32, i32 @callee_func () {
init:
  bid: b0
  allocations:
unresolved:

block b0:
  ret 15:i32, 20:i32
}

fun unit @caller_func () {
init:
  bid: b0
  allocations:
unresolved:
  %U0:i32 = unresolved
  %U1:i32 = unresolved

block b0:
  %b0:i0:i32, %b0:i1:i32 = call @callee_func:[ret:i32, i32 params:()]*()
  %b0:i2:i32 = add %U0:i32 %U1:i32
  ret %b0:i2:i32
}
)");

    /// resolve the unresolved IRInstruction
    /// U0 -> %b0:i0:i32
    /// U1 -> %b0:i1:i32

    auto module = Module::create(std::move(ir));
    module->replaceInst(u0.getStorage(), call.getResult(0), true);
    module->replaceInst(u1.getStorage(), call.getResult(1), true);

    result.clear();
    module->getIR()->print(printContext);
    STR_EQ(result, R"(
constants:
  15:i32
  20:i32
  @callee_func:[ret:i32, i32 params:()]*

fun i32, i32 @callee_func () {
init:
  bid: b0
  allocations:
unresolved:

block b0:
  ret 15:i32, 20:i32
}

fun unit @caller_func () {
init:
  bid: b0
  allocations:
unresolved:

block b0:
  %b0:i0:i32, %b0:i1:i32 = call @callee_func:[ret:i32, i32 params:()]*()
  %b0:i2:i32 = add %b0:i0:i32 %b0:i1:i32
  ret %b0:i2:i32
})");
  }

  SUBCASE("Replace exit") {
    auto ir = std::make_unique<IR>(&context);

    auto i32 = IntT::get(&context, 32, true);
    auto funcType = FunctionT::get(&context, i32, {});

    Function *func = new Function("test_func", funcType, ir.get(), &context);

    ir->addFunction(func);

    Block *block0 = func->addBlock(0);
    func->setEntryBlock(0);

    Block *block1 = func->addBlock(1);
    Block *block2 = func->addBlock(2);

    IRBuilder builder(&context);
    builder.setInsertionPoint(ir->getConstantBlock());

    auto const0 = builder.create<inst::Constant>(
        {}, ConstantIntAttr::get(&context, 0, 1, false));

    auto const15 = builder.create<inst::Constant>(
        {}, ConstantIntAttr::get(&context, 15, 32, true));

    auto const30 = builder.create<inst::Constant>(
        {}, ConstantIntAttr::get(&context, 30, 32, true));

    builder.setInsertionPoint(block0);

    auto br = builder.create<inst::Branch>({}, const0, JumpArgState(block1),
                                           JumpArgState(block2));

    builder.setInsertionPoint(block1);

    auto ret15 = builder.create<inst::Return>({}, const15);

    builder.setInsertionPoint(block2);

    auto ret30 = builder.create<inst::Return>({}, const30);

    std::string result;
    llvm::raw_string_ostream ss(result);
    IRPrintContext printContext(ss, IRPrintContext::Default);

    ir->print(printContext);
    STR_EQ(result, R"(
fun i32 @test_func () {
init:
  bid: b0
  allocations:

block b0:
  br 0:u1, b1(), b2()

block b1:
  ret 15:i32

block b2:
  ret 30:i32
}
)");

    auto module = Module::create(std::move(ir));

    auto b0_Successors_before = module->getSuccessors(block0);

    CHECK(b0_Successors_before.size() == 2);
    CHECK(b0_Successors_before.contains(block1));
    CHECK(b0_Successors_before.contains(block2));

    auto b1_Predecessors_before = module->getPredecessors(block1);
    CHECK(b1_Predecessors_before.size() == 1);
    CHECK(*b1_Predecessors_before.begin() == block0);

    auto b2_Predecessors_before = module->getPredecessors(block2);
    CHECK(b2_Predecessors_before.size() == 1);
    CHECK(*b2_Predecessors_before.begin() == block0);

    for (Block *block : *func) {
      module->replaceExit(
          block->getExit(),
          [&](IRBuilder &builder, BlockExit oldExit) -> BlockExit {
            if (auto branch = oldExit.dyn_cast<inst::Branch>()) {
              auto value = branch.getCondition();
              auto constantVal =
                  value.getInstruction()->getDefiningInst<inst::Constant>();

              if (!constantVal)
                return nullptr;

              auto intVal = constantVal.getValue().dyn_cast<ConstantIntAttr>();
              if (!intVal)
                return nullptr;

              assert(!intVal.getIntType().isSigned() &&
                     intVal.getBitWidth() == 1);

              auto jumpArg = intVal.getValue() == 0 ? branch.getElseArg()
                                                    : branch.getIfArg();
              BlockExit newExit = builder.create<inst::Jump>(
                  branch.getRange(), jumpArg->getAsState());
              return newExit;
            }
            return nullptr;
          });
    }

    result.clear();
    module->getIR()->print(printContext);
    STR_EQ(result, R"(
fun i32 @test_func () {
init:
  bid: b0
  allocations:

block b0:
  j b2()

block b1:
  ret 15:i32

block b2:
  ret 30:i32
}
)");

    auto b0_Successors_after = module->getSuccessors(block0);

    CHECK(b0_Successors_after.size() == 1);
    CHECK(*b0_Successors_after.begin() == block2);

    auto b1_Predecessors_after = module->getPredecessors(block1);
    CHECK(b1_Predecessors_after.size() == 0);

    auto b2_Predecessors_after = module->getPredecessors(block2);
    CHECK(b2_Predecessors_after.size() == 1);
    CHECK(*b2_Predecessors_after.begin() == block0);
  }
}

} // namespace kecc::ir
