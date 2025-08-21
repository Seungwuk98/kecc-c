#include "TestUtils.h"
#include "kecc/translate/MoveSchedule.h"
#include "llvm/ADT/ArrayRef.h"

namespace kecc {

TEST_CASE("Move schedule") {

  SUBCASE("Normal case") {
    // 1 <- 2 <- 3 <- 4
    llvm::ArrayRef<int> dst = {1, 2, 3};
    llvm::ArrayRef<int> src = {2, 3, 4};
    llvm::ArrayRef<std::tuple<Movement, int, int>> expected = {
        {Movement::Move, 1, 2},
        {Movement::Move, 2, 3},
        {Movement::Move, 3, 4},
    };

    MoveManagement<int> moveManagement(dst, src, 0);
    auto moveSchedule = moveManagement.getMoveSchedule();
    CHECK_EQ(moveSchedule.size(), expected.size());
    CHECK_EQ(moveSchedule, expected);
  }

  SUBCASE("Normal case 2") {
    // 1 <- 2
    // 2 <- 4
    // 3 <- 3

    llvm::ArrayRef<int> dst = {1, 2, 3};
    llvm::ArrayRef<int> src = {2, 4, 3};

    llvm::ArrayRef<std::tuple<Movement, int, int>> expected = {
        {Movement::Move, 1, 2},
        {Movement::Move, 2, 4},
    };

    MoveManagement<int> moveManagement(dst, src, 0);
    auto moveSchedule = moveManagement.getMoveSchedule();
    CHECK_EQ(moveSchedule.size(), expected.size());
    CHECK_EQ(moveSchedule, expected);
  }

  SUBCASE("Loop case") {
    // 1 <- 2
    // 2 <- 1
    // 3 <- 4
    // 4 <- 3
    // 5 <- 6

    llvm::ArrayRef<int> dst = {1, 2, 3, 4, 5};
    llvm::ArrayRef<int> src = {2, 1, 4, 3, 6};

    llvm::ArrayRef<std::tuple<Movement, int, int>> expected = {
        {Movement::Swap, 1, 2},
        {Movement::Swap, 3, 4},
        {Movement::Move, 5, 6},
    };

    MoveManagement<int> moveManagement(dst, src, 0);
    auto moveSchedule = moveManagement.getMoveSchedule();
    CHECK_EQ(moveSchedule.size(), expected.size());
    CHECK_EQ(moveSchedule, expected);
  }

  SUBCASE("Loop case 2") {
    // 1 <- 2
    // 2 <- 3
    // 3 <- 1  ... 3 <- 0
    // 4 <- 5
    // 5 <- 6

    llvm::ArrayRef<int> dst = {1, 2, 3, 4, 5};
    llvm::ArrayRef<int> src = {2, 3, 1, 5, 6};
    llvm::ArrayRef<std::tuple<Movement, int, int>> expected = {
        {Movement::Move, 4, 5}, {Movement::Move, 5, 6}, {Movement::Move, 0, 1},
        {Movement::Move, 1, 2}, {Movement::Move, 2, 3}, {Movement::Move, 3, 0},
    };

    MoveManagement<int> moveManagement(dst, src, 0);
    auto moveSchedule = moveManagement.getMoveSchedule();
    CHECK_EQ(moveSchedule.size(), expected.size());
    CHECK_EQ(moveSchedule, expected);
  }

  SUBCASE("Loop case 3") {
    // 1 <- 2
    // 2 <- 3
    // 3 <- 1  .. 3 <- 0
    // 4 <- 5
    // 5 <- 6
    // 6 <- 4  .. 6 <- 0

    llvm::ArrayRef<int> dst = {1, 2, 3, 4, 5, 6};
    llvm::ArrayRef<int> src = {2, 3, 1, 5, 6, 4};
    llvm::ArrayRef<std::tuple<Movement, int, int>> expected = {
        {Movement::Move, 0, 1}, {Movement::Move, 1, 2}, {Movement::Move, 2, 3},
        {Movement::Move, 3, 0}, {Movement::Move, 0, 4}, {Movement::Move, 4, 5},
        {Movement::Move, 5, 6}, {Movement::Move, 6, 0},
    };

    MoveManagement<int> moveManagement(dst, src, 0);
    auto moveSchedule = moveManagement.getMoveSchedule();
    CHECK_EQ(moveSchedule.size(), expected.size());
    CHECK_EQ(moveSchedule, expected);
  }
}

} // namespace kecc
