#include "TestTensor.h"
#include "TestUtils.h"
#include "kecc/ir/Module.h"
#include "kecc/ir/PatternMatch.h"

namespace kecc::ir {
TEST_CASE("Tensor Type") {
  IRContext context;
  context.registerType<TensorT>();
  context.registerInst<inst::CreateTensor>();
  context.registerInst<inst::Transpose>();

  SUBCASE("Create Tensor Type") {
    Type intT = IntT::get(&context, 32, true);
    TensorT tensorT = TensorT::get(&context, intT, {2, 3, 4});

    TensorT tensorT2 = TensorT::get(&context, intT, {2, 3, 4});
    CHECK(tensorT == tensorT2);

    CHECK(tensorT.toString() == "tensor<i32, (2, 3, 4)>");
  }

  SUBCASE("IR Create Tensor Type") {
    std::unique_ptr<IR> ir = std::make_unique<IR>(&context);

    IntT intT = IntT::get(&context, 32, true);
    TensorT tensorT = TensorT::get(&context, intT, {1, 3});
    FunctionT funcT = FunctionT::get(&context, {tensorT}, {});

    Function *func = new Function("tensor_func", funcT, ir.get(), &context);
    ir->addFunction(func);

    Block *block = func->addBlock(0);
    func->setEntryBlock(0);

    IRBuilder builder(&context);

    builder.setInsertionPoint(ir->getConstantBlock());
    auto const1 = builder.create<inst::Constant>(
        {}, ConstantIntAttr::get(&context, 1, 32, true));
    auto const2 = builder.create<inst::Constant>(
        {}, ConstantIntAttr::get(&context, 2, 32, true));
    auto const3 = builder.create<inst::Constant>(
        {}, ConstantIntAttr::get(&context, 3, 32, true));

    builder.setInsertionPoint(block);

    llvm::ArrayRef<size_t> shape = {1, 3};
    llvm::ArrayRef<Value> operands = {const1, const2, const3};
    inst::CreateTensor tensor =
        builder.create<inst::CreateTensor>({}, shape, operands);

    CHECK(tensor.getType().isa<TensorT>());
    CHECK(tensor.getType() == tensorT);
    CHECK(tensor.getElementType() == intT);

    inst::Transpose transpose = builder.create<inst::Transpose>({}, tensor);
    inst::Transpose transpose2 = builder.create<inst::Transpose>({}, transpose);

    builder.create<inst::Return>({}, transpose2);

    std::string result;
    llvm::raw_string_ostream os(result);
    IRPrintContext printContext(os);
    ir->print(printContext);

    STR_EQ(result, R"(
fun tensor<i32, (1, 3)> @tensor_func () {
init:
  bid: b0
  allocations:

block b0:
  %b0:i0:tensor<i32, (1, 3)> = tensor [1:i32, 2:i32, 3:i32]
  %b0:i1:tensor<i32, (3, 1)> = transpose %b0:i0:tensor<i32, (1, 3)>
  %b0:i2:tensor<i32, (1, 3)> = transpose %b0:i1:tensor<i32, (3, 1)>
  ret %b0:i2:tensor<i32, (1, 3)>
}
)");

    auto module = Module::create(std::move(ir));

    PatternSet set;
    addTransposePattern(set);
    applyPatternConversion(module.get(), set);

    result.clear();
    module->getIR()->print(printContext);

    STR_EQ(result, R"(
fun tensor<i32, (1, 3)> @tensor_func () {
init:
  bid: b0
  allocations:

block b0:
  %b0:i0:tensor<i32, (1, 3)> = tensor [1:i32, 2:i32, 3:i32]
  %b0:i1:tensor<i32, (3, 1)> = transpose %b0:i0:tensor<i32, (1, 3)>
  ret %b0:i0:tensor<i32, (1, 3)>
}
)");
  }
}

} // namespace kecc::ir
