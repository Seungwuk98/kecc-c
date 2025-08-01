#include "TestUtils.h"
#include "kecc/ir/IR.h"
#include "kecc/ir/IRInstructions.h"

namespace kecc::ir {

TEST_CASE("Usage Tests") {
  IRContext context;

  SUBCASE("Def-Use") {
    auto ir = std::make_unique<IR>(&context);

    auto unitT = UnitT::get(&context);
    auto i32 = IntT::get(&context, 32, true);
    auto funcType = FunctionT::get(&context, unitT, {i32});

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
    Value const25 = builder.create<inst::Constant>(
        {}, ConstantIntAttr::get(&context, 25, 32, true));

    builder.setInsertionPoint(block);
    auto add =
        builder.create<inst::Binary>({}, const15, const20, inst::Binary::Add);

    auto sub =
        builder.create<inst::Binary>({}, const20, const15, inst::Binary::Sub);

    auto ret = builder.create<inst::Return>({}, sub);

    std::string result;
    llvm::raw_string_ostream ss(result);
    IRPrintContext printContext(ss, IRPrintContext::Debug);
    ir->print(printContext);

    STR_EQ(result, R"(
constants:
  15:i32
  20:i32
  25:i32

fun unit @test_func (i32) {
init:
  bid: b0
  allocations:
unresolved:

block b0:
  %b0:i0:i32 = add 15:i32 20:i32
  %b0:i1:i32 = sub 20:i32 15:i32
  ret %b0:i1:i32
}
)");

    auto const15_useBegin = const15.useBegin();
    Operand *const15_first = *const15_useBegin;
    CHECK(const15_first->getOwner() == add.getStorage());
    Operand *const15_second = *(++const15_useBegin);
    CHECK(const15_second->getOwner() == sub.getStorage());

    auto const20_useBegin = const20.useBegin();
    Operand *const20_first = *const20_useBegin;
    CHECK(const20_first->getOwner() == add.getStorage());
    Operand *const20_second = *(++const20_useBegin);
    CHECK(const20_second->getOwner() == sub.getStorage());

    auto const25_useBegin = const25.useBegin();
    CHECK(const25_useBegin == const25.useEnd());

    // replace 15 to 25
    Value const15V = const15;
    const15V.replaceWith(const25);

    result.clear();
    ir->print(printContext);
    STR_EQ(result, R"(
constants:
  15:i32
  20:i32
  25:i32

fun unit @test_func (i32) {
init:
  bid: b0
  allocations:
unresolved:

block b0:
  %b0:i0:i32 = add 25:i32 20:i32
  %b0:i1:i32 = sub 20:i32 25:i32
  ret %b0:i1:i32
}
)");
  }
}

} // namespace kecc::ir
