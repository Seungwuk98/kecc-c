#include "TestUtils.h"
#include "doctest/doctest.h"
#include "kecc//ir/IR.h"
#include "kecc/ir/Context.h"
#include "kecc/ir/IRBuilder.h"
#include "kecc/ir/IRInstructions.h"
#include "kecc/ir/IRTypes.h"

namespace kecc::ir {
TEST_CASE("IR Tests") {
  IRContext context;

  SUBCASE("IR") {
    auto ir = std::make_unique<IR>(&context);

    auto unitT = UnitT::get(&context);
    auto i32 = IntT::get(&context, 32, true);

    auto funcType = FunctionT::get(&context, unitT, {i32, i32});

    Function *func = new Function("test_func", funcType, ir.get(), &context);

    ir->addFunction(func);
    Function *retrievedFunc = ir->getFunction("test_func");
    CHECK(retrievedFunc == func);

    CHECK(ir->getFunctionCount() == 1);

    ir->erase("test_func");
    CHECK(ir->getFunctionCount() == 0);
  }

  SUBCASE("Block") {
    auto ir = std::make_unique<IR>(&context);

    auto unitT = UnitT::get(&context);
    auto i32 = IntT::get(&context, 32, true);

    auto funcType = FunctionT::get(&context, unitT, {i32, i32});

    Function *func = new Function("test_func", funcType, ir.get(), &context);

    ir->addFunction(func);
    Block *block = func->addBlock(0);
    func->setEntryBlock(0);

    IRBuilder builder(&context);
    builder.setInsertionPoint(block);

    auto phi = builder.create<Phi>({}, i32);
    phi.setValueName("test");
    auto ret = builder.create<inst::Return>({});

    auto it = block->begin();
    auto firstInst = *(it++);
    auto secondInst = *(it++);

    CHECK(phi.getStorage() == firstInst);
    CHECK(ret.getStorage() == secondInst);
    CHECK(firstInst->getDefiningInst<Phi>() == phi);
    CHECK(secondInst->getDefiningInst<inst::Return>() == ret);
    CHECK(it == block->end());

    std::string result;
    llvm::raw_string_ostream ss(result);
    IRPrintContext printContext(ss, IRPrintContext::Default);

    ir->print(printContext);
    STR_EQ(result, R"(

fun unit @test_func (i32, i32) {
init:
  bid: b0
  allocations:

block b0:
  %b0:p0:i32:test
  ret
}

)");

    builder.setInsertionPoint(func->getAllocationBlock());
    auto localVar =
        builder.create<inst::LocalVariable>({}, PointerT::get(&context, i32));
    localVar.setValueName("local_var");

    auto allocIt = func->getAllocationBlock()->begin();
    auto allocInst = (*allocIt)->getDefiningInst<inst::LocalVariable>();
    CHECK(allocInst == localVar);

    result.clear();
    ir->print(printContext);

    STR_EQ(result, R"(

fun unit @test_func (i32, i32) {
init:
  bid: b0
  allocations:
    %l0:i32:local_var

block b0:
  %b0:p0:i32:test
  ret
}

)");

    it = block->begin();
    builder.setInsertionPoint(Block::InsertionPoint(block, it));
    auto load = builder.create<inst::Load>({}, localVar);
    CHECK(load == (*++it)->getDefiningInst<inst::Load>());

    result.clear();
    ir->print(printContext);

    STR_EQ(result, R"(

fun unit @test_func (i32, i32) {
init:
  bid: b0
  allocations:
    %l0:i32:local_var

block b0:
  %b0:p0:i32:test
  %b0:i0:i32 = load %l0:i32*
  ret
}
)");
  }

  SUBCASE("Global Variable") {
    auto ir = std::make_unique<IR>(&context);

    auto unitT = UnitT::get(&context);
    auto i32 = IntT::get(&context, 32, true);

    IRBuilder builder(&context);
    builder.setInsertionPoint(ir->getGlobalBlock());

    auto globalVar =
        builder.create<inst::GlobalVariableDefinition>({}, i32, "global_var");

    auto funcT = FunctionT::get(&context, unitT, {i32, i32});
    auto func = new Function("test_func", funcT, ir.get(), &context);

    ir->addFunction(func);

    func->addBlock(0);
    func->setEntryBlock(0);

    builder.setInsertionPoint(ir->getConstantBlock());
    auto variable = ConstantVariableAttr::get(&context, "global_var",
                                              PointerT::get(&context, i32));

    auto variableMem = builder.create<inst::Constant>({}, variable);

    builder.setInsertionPoint(func->getEntryBlock());
    auto loadGlobal = builder.create<inst::Load>({}, variableMem);

    auto ret = builder.create<inst::Return>({});

    std::string result;
    llvm::raw_string_ostream ss(result);
    IRPrintContext printContext(ss, IRPrintContext::Default);

    ir->print(printContext);
    STR_EQ(result, R"(
var i32 @global_var

fun unit @test_func (i32, i32) {
init:
  bid: b0
  allocations:

block b0:
  %b0:i0:i32 = load @global_var:i32*
  ret
}
)");
  }

  SUBCASE("Struct") {
    auto ir = std::make_unique<IR>(&context);
    auto unitT = UnitT::get(&context);
    auto i32 = IntT::get(&context, 32, true);

    IRBuilder builder(&context);
    builder.setInsertionPoint(ir->getStructBlock());

    llvm::SmallVector<std::pair<llvm::StringRef, Type>> fields = {
        {"field1", i32},
        {"field2", i32},
    };

    auto structDef =
        builder.create<inst::StructDefinition>({}, fields, "TestStruct");

    std::string result;
    llvm::raw_string_ostream ss(result);
    IRPrintContext printContext(ss, IRPrintContext::Default);

    ir->print(printContext);

    STR_EQ(result, R"(
struct TestStruct : { field1:i32, field2:i32 }
)");

    fields.clear();
    auto sturctDef2 =
        builder.create<inst::StructDefinition>({}, fields, "OpaqueStruct");

    result.clear();
    ir->print(printContext);

    STR_EQ(result, R"(
struct TestStruct : { field1:i32, field2:i32 }
struct OpaqueStruct : opaque
)");
  }

  SUBCASE("Constant") {
    auto ir = std::make_unique<IR>(&context);
    auto unitT = UnitT::get(&context);
    auto i32 = IntT::get(&context, 32, true);

    IRBuilder builder(&context);
    builder.setInsertionPoint(ir->getConstantBlock());

    auto const15 = builder.create<inst::Constant>(
        {}, ConstantIntAttr::get(&context, 15, 32, true));

    Function *func = new Function(
        "test_func", FunctionT::get(&context, i32, {}), ir.get(), &context);

    ir->addFunction(func);

    Block *block = func->addBlock(0);
    func->setEntryBlock(0);

    builder.setInsertionPoint(block);
    auto add =
        builder.create<inst::Binary>({}, const15, const15, inst::Binary::Add);

    auto ret = builder.create<inst::Return>({}, add);

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
  %b0:i0:i32 = add 15:i32 15:i32
  ret %b0:i0:i32
}
)");

    result.clear();
    printContext.setMode(IRPrintContext::Debug);
    ir->print(printContext);

    STR_EQ(result, R"(
constants:
  15:i32

fun i32 @test_func () {
init:
  bid: b0
  allocations:
unresolved:

block b0:
  %b0:i0:i32 = add 15:i32 15:i32
  ret %b0:i0:i32
}
)");
  }

  SUBCASE("Multi Block") {
    auto ir = std::make_unique<IR>(&context);
    auto unitT = UnitT::get(&context);
    auto i32 = IntT::get(&context, 32, true);

    IRBuilder builder(&context);
    builder.setInsertionPoint(ir->getConstantBlock());

    auto const15 = builder.create<inst::Constant>(
        {}, ConstantIntAttr::get(&context, 15, 32, true));
    auto const25 = builder.create<inst::Constant>(
        {}, ConstantIntAttr::get(&context, 25, 32, true));

    Function *func = new Function(
        "test_func", FunctionT::get(&context, i32, {}), ir.get(), &context);

    ir->addFunction(func);

    Block *block = func->addBlock(0);
    Block *block2 = func->addBlock(1);
    func->setEntryBlock(0);

    builder.setInsertionPoint(block);

    auto add =
        builder.create<inst::Binary>({}, const15, const25, inst::Binary::Add);

    builder.setInsertionPoint(block2);

    auto phi = builder.create<Phi>({}, i32);
    auto ret = builder.create<inst::Return>({}, phi);

    builder.setInsertionPoint(block);

    auto jump = builder.create<inst::Jump>({}, JumpArgState(block2, add));

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
  %b0:i0:i32 = add 15:i32 25:i32
  j b1(%b0:i0:i32)

block b1:
  %b1:p0:i32
  ret %b1:p0:i32
})");
  }
}
} // namespace kecc::ir
