#include "doctest/doctest.h"
#include "kecc/ir/IRAttributes.h"

namespace kecc::ir {

TEST_CASE("IR Attrs") {
  IRContext context;

  SUBCASE("Attr build") {
    auto strAttr = StringAttr::get(&context, "test");
    CHECK_EQ(strAttr.getValue(), "test");
  }

  SUBCASE("Attr equality") {
    auto strAttr1 = StringAttr::get(&context, "test");
    auto strAttr2 = StringAttr::get(&context, "test");
    auto strAttr3 = StringAttr::get(&context, "different");

    CHECK(strAttr1 == strAttr2);
    CHECK(strAttr1 != strAttr3);
    CHECK(strAttr2 != strAttr3);
  }

  SUBCASE("Constant Int Attr") {
    auto intAttr = ConstantIntAttr::get(&context, 10, 32, true);

    CHECK_EQ(intAttr.getValue(), 10);
    CHECK_EQ(intAttr.getBitWidth(), 32);
    CHECK(intAttr.isSigned());

    auto intAttrCopy = ConstantIntAttr::get(&context, 10, 32, true);
    CHECK(intAttr == intAttrCopy);
  }

  SUBCASE("Constant Float Attr") {
    auto floatAttr = ConstantFloatAttr::get(&context, 3.14f);

    CHECK_EQ(floatAttr.getValue(), llvm::APFloat(3.14f));

    auto floatAttrCopy = ConstantFloatAttr::get(&context, 3.14f);
    CHECK(floatAttr == floatAttrCopy);
  }

  SUBCASE("Array Attr") {
    auto intAttr = ConstantIntAttr::get(&context, 42, 64, true);
    auto strAttr = StringAttr::get(&context, "example");
    auto arrayAttr = ArrayAttr::get(&context, {intAttr, strAttr});

    CHECK_EQ(arrayAttr.getValues().size(), 2);
    CHECK(arrayAttr.getValues()[0] == intAttr);
    CHECK(arrayAttr.getValues()[1] == strAttr);

    auto arrayAttrCopy = ArrayAttr::get(&context, {intAttr, strAttr});
    CHECK(arrayAttr == arrayAttrCopy);
  }
}

TEST_CASE("Attr walks") {
  IRContext context_;
  IRContext *context = &context_;

  SUBCASE("Type attr walk") {
    auto i32 = IntT::get(context, 32, true);
    auto f32 = FloatT::get(context, 32);
    auto i32_f32 = TupleT::get(context, {i32, f32});

    auto attr = TypeAttr::get(context, i32_f32);

    llvm::SmallVector<Type> visit;
    attr.walk([&](Type type) -> WalkResult {
      visit.push_back(type);
      return WalkResult::advance();
    });

    CHECK_EQ(visit.size(), 3);
    CHECK(visit[0] == i32_f32);
    CHECK(visit[1] == i32);
    CHECK(visit[2] == f32);
  }

  SUBCASE("Type attr replace") {
    auto i32 = IntT::get(context, 32, true);
    auto f32 = FloatT::get(context, 32);
    auto i32_f32 = TupleT::get(context, {i32, f32});
    auto i32_f32__i32_f32 = TupleT::get(context, {i32_f32, i32_f32});

    auto attr = TypeAttr::get(context, i32_f32__i32_f32);

    auto replaced = attr.replace([&](Type type) -> ReplaceResult<Type> {
      if (auto floatT = type.dyn_cast<FloatT>()) {
        return {PointerT::get(context, floatT),
                utils::LogicalResult::success()};
      }
      return {type, utils::LogicalResult::success()};
    });
    CHECK(replaced);

    auto pointerF32 = PointerT::get(context, f32);
    auto i32_f32Ptr = TupleT::get(context, {i32, pointerF32});
    auto expectedType = TupleT::get(context, {i32_f32Ptr, i32_f32Ptr});
    auto expectedAttr = TypeAttr::get(context, expectedType);
    CHECK_EQ(replaced, expectedAttr);
  }
}

} // namespace kecc::ir
