#include "doctest/doctest.h"
#include "kecc/ir/IRTypes.h"
#include "kecc/ir/WalkSupport.h"
#include "kecc/utils/LogicalResult.h"

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

TEST_CASE("Type walk") {
  IRContext context_;
  IRContext *context = &context_;

  SUBCASE("Walk") {
    auto i32 = IntT::get(context, 32, true);
    auto f32 = FloatT::get(context, 32);
    auto i32_f32 = TupleT::get(context, {i32, f32});

    llvm::SmallVector<Type> visit;
    i32_f32.walk([&](Type type) -> WalkResult {
      visit.push_back(type);
      return WalkResult::advance();
    });

    CHECK_EQ(visit.size(), 3);
    CHECK_EQ(visit[0], i32_f32);
    CHECK_EQ(visit[1], i32);
    CHECK_EQ(visit[2], f32);

    visit.clear();

    i32_f32.walk<TypeWalker::PostOrder>([&](Type type) -> WalkResult {
      visit.push_back(type);
      return WalkResult::advance();
    });

    CHECK_EQ(visit.size(), 3);
    CHECK_EQ(visit[0], i32);
    CHECK_EQ(visit[1], f32);
    CHECK_EQ(visit[2], i32_f32);
  }

  SUBCASE("Replace") {
    auto i32 = IntT::get(context, 32, true);
    auto f32 = FloatT::get(context, 32);
    auto structX = NameStruct::get(context, "X");
    auto structY = NameStruct::get(context, "Y");
    auto funcT = FunctionT::get(context, {structX, structY}, {i32, structY});

    auto func2T = FunctionT::get(context, {structY, funcT},
                                 {f32, PointerT::get(context, funcT)});

    auto replaceFn = [&](Type type) -> ReplaceResult<Type> {
      if (auto funcT = type.dyn_cast<FunctionT>()) {
        llvm::SmallVector<Type> retTypes;
        llvm::SmallVector<Type> argTypes;

        for (auto retType : funcT.getReturnTypes()) {
          if (retType.isa<NameStruct>()) {
            argTypes.emplace_back(PointerT::get(context, retType));
          } else {
            retTypes.emplace_back(retType);
          }
        }

        for (auto argType : funcT.getArgTypes()) {
          if (argType.isa<NameStruct>())
            argType = PointerT::get(context, argType);
          argTypes.emplace_back(argType);
        }

        if (retTypes.empty())
          retTypes.emplace_back(UnitT::get(context));
        return {FunctionT::get(context, retTypes, argTypes),
                utils::LogicalResult::success()};
      }
      return {type, utils::LogicalResult::success()};
    };

    auto replaced = func2T.replace(replaceFn);

    auto [replacedFuncT, result] = replaceFn(funcT);

    Type replacedFunc2T =
        FunctionT::get(context, {structY, replacedFuncT},
                       {f32, PointerT::get(context, replacedFuncT)});
    replacedFunc2T = replaceFn(replacedFunc2T).first;

    CHECK_EQ(replaced, replacedFunc2T);
  }
}
} // namespace kecc::ir
