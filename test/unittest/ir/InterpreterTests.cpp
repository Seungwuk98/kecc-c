#include "TestUtils.h"
#include "kecc/ir/Interpreter.h"
#include "kecc/parser/Parser.h"

namespace kecc::ir {

TEST_CASE("Interpreter Tests") {
  IRContext context;

  SUBCASE("Simple execute") {
    auto code = R"ir(
fun i32 @main() {
init:
  bid: b0
  allocations:

block b0:
  ret 15:i32
}
)ir";

    auto buffer = llvm::MemoryBuffer::getMemBuffer(code, "test_ir");
    auto index = context.getSourceMgr().AddNewSourceBuffer(std::move(buffer),
                                                           llvm::SMLoc());
    auto bufferRef = context.getSourceMgr().getMemoryBuffer(index);
    Lexer lexer(bufferRef->getBuffer(), &context);
    Parser parser(lexer, &context);

    auto module = parser.parseAndBuildModule();
    REQUIRE_FALSE(context.diag().hasError());

    Interpreter interpreter(module.get());

    auto result = interpreter.call("main", {});
    CHECK(result.size() == 1);

    VRegister *vReg = result[0].get();
    CHECK(vReg->isa<VRegisterInt>());
    auto intReg = vReg->cast<VRegisterInt>();
    CHECK(intReg->getValue() == 15);
  }
}

} // namespace kecc::ir
