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

} // namespace kecc::ir
