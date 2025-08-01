#include "TestUtils.h"
#include "doctest/doctest.h"
#include "kecc/asm/Asm.h"
#include "kecc/asm/AsmBuilder.h"
#include "kecc/asm/AsmInstruction.h"

namespace kecc::as {

TEST_CASE("ASM Tests") {
  SUBCASE("ASM Block") {
    Block *block = new Block("test_block");
    AsmBuilder builder;
    builder.setInsertionPointLast(block);

    rtype::Add *add = builder.create<rtype::Add>(
        Register::a0(), Register::a0(), Register::a1(), DataSize::word());

    rtype::Sub *sub = builder.create<rtype::Sub>(
        Register::a0(), Register::a0(), Register::a1(), DataSize::word());

    pseudo::Ret *ret = builder.create<pseudo::Ret>();

    std::string result;
    llvm::raw_string_ostream os(result);
    block->print(os);

    STR_EQ(result, "test_block:\n"
                   "  addw\ta0,a0,a1\n"
                   "  subw\ta0,a0,a1\n"
                   "  ret\n");

    sub->remove();
    result.clear();
    block->print(os);

    STR_EQ(result, "test_block:\n"
                   "  addw\ta0,a0,a1\n"
                   "  ret\n");

    delete block;
  }
}

} // namespace kecc::as
