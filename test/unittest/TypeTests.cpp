#include "doctest/doctest.h"
#include "kecc/ir/IRTypes.h"

namespace kecc::ir {
TEST_CASE("IR Types") {
  IRContext context;

  SUBCASE("Type Build") {
    auto i2 = IntT::get(&context, 2, true);
    CHECK_EQ(i2.toString(), "i2");

    auto u2 = IntT::get(&context, 2, false);
    CHECK_EQ(u2.toString(), "u2");
  }

  SUBCASE("Type Equality") {
    auto i2 = IntT::get(&context, 2, true);
    auto i2Copy = IntT::get(&context, 2, true);
    auto u2 = IntT::get(&context, 2, false);
    auto u3 = IntT::get(&context, 3, false);
    CHECK(i2 == i2Copy);
    CHECK(i2 != u2);
    CHECK(u2 != u3);
    CHECK(i2 != u3);
  }

  SUBCASE("Type Equality - function") {
    auto i2 = IntT::get(&context, 2, true);
    auto i2Pointer = PointerT::get(&context, i2);
    auto unit = UnitT::get(&context);

    auto function = FunctionT::get(&context, unit, {i2, i2Pointer});
    CHECK_EQ(function.getReturnTypes()[0], unit);
    CHECK_EQ(function.getArgTypes().size(), 2);
    CHECK_EQ(function.getArgTypes()[0], i2);
    CHECK_EQ(function.getArgTypes()[1], i2Pointer);
    CHECK_EQ(function.toString(), "[ret:unit params:(i2, i2*)]");

    auto functionCopy = FunctionT::get(&context, unit, {i2, i2Pointer});
    CHECK(function == functionCopy);
  }

  SUBCASE("Type Equality - pointer") {
    auto i32 = IntT::get(&context, 32, false);
    auto i32Pointer = PointerT::get(&context, i32);

    auto i32PointerCopy = PointerT::get(&context, i32);

    CHECK(i32Pointer == i32PointerCopy);
  }
}
} // namespace kecc::ir
