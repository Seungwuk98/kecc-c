#include "TestUtils.h"
#include "kecc/ir/Interpreter.h"
#include "kecc/parser/Parser.h"

namespace kecc::ir {

llvm::SmallVector<std::unique_ptr<VRegister>>
runFunction(IRContext *context, llvm::StringRef code, llvm::StringRef funcName,
            llvm::ArrayRef<VRegister *> args) {
  auto buffer = llvm::MemoryBuffer::getMemBuffer(code, "test_ir");
  auto index = context->getSourceMgr().AddNewSourceBuffer(std::move(buffer),
                                                          llvm::SMLoc());
  auto bufferRef = context->getSourceMgr().getMemoryBuffer(index);
  Lexer lexer(bufferRef->getBuffer(), context);
  Parser parser(lexer, context);
  auto module = parser.parseAndBuildModule();
  REQUIRE_FALSE(context->diag().hasError());
  Interpreter interpreter(module.get());
  return interpreter.call(funcName, args);
}

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

    auto result = runFunction(&context, code, "main", {});

    CHECK(result.size() == 1);

    VRegister *vReg = result[0].get();
    CHECK(vReg->isa<VRegisterInt>());
    auto intReg = vReg->cast<VRegisterInt>();
    CHECK(intReg->getValue() == 15);
  }

  SUBCASE("Simple execute 2") {
    auto code = R"ir(
fun i32 @func(i32) {
init:
  bid: b0
  allocations:
    %l0:i32:a

block b0:
  %b0:p0:i32:a
  ret %b0:p0:i32
}
    )ir";

    auto arg = std::make_unique<VRegisterInt>(42);
    auto result = runFunction(&context, code, "func", {arg.get()});
    CHECK(result.size() == 1);
    VRegister *vReg = result[0].get();
    CHECK(vReg->isa<VRegisterInt>());
    auto intReg = vReg->cast<VRegisterInt>();
    CHECK(intReg->getValue() == 42);
  }

  SUBCASE("Recursive execute") {
    auto code = R"ir(
fun i32 @sum(i32) {
init:
  bid: b0
  allocations:

block b0:
  %b0:p0:i32:n
  %b0:i0:i1 = cmp lt %b0:p0:i32 10:i32
  br %b0:i0:i1, b1(), b2()

block b1:
  %b1:i0:i32 = add %b0:p0:i32 1:i32
  %b1:i1:i32 = call @sum:[ret:i32 params:(i32)]*(%b1:i0:i32)
  %b1:i2:i32 = add %b0:p0:i32 %b1:i1:i32
  ret %b1:i2:i32

block b2:
  ret 0:i32 
}
)ir";

    auto arg = std::make_unique<VRegisterInt>(0);
    auto result = runFunction(&context, code, "sum", {arg.get()});
    CHECK(result.size() == 1);
    VRegister *vReg = result[0].get();
    CHECK(vReg->isa<VRegisterInt>());
    auto intReg = vReg->cast<VRegisterInt>();
    CHECK(intReg->getValue() == 45);
  }

  SUBCASE("global variable") {
    auto code = R"ir(
var i32 @g = 100

fun i32 @main() {
init:
  bid: b0
  allocations:

block b0:
  %b0:i0:i32 = load @g:i32*
  %b0:i1:i32 = add %b0:i0:i32 23:i32
  %b0:i2:unit = store %b0:i1:i32 @g:i32*
  %b0:i3:i32 = load @g:i32*
  ret %b0:i3:i32
}
  )ir";

    auto result = runFunction(&context, code, "main", {});
    CHECK(result.size() == 1);
    VRegister *vReg = result[0].get();
    CHECK(vReg->isa<VRegisterInt>());
    auto intReg = vReg->cast<VRegisterInt>();
    CHECK(intReg->getValue() == 123);
  }

  SUBCASE("global struct test") {
    auto code = R"ir(
struct S : { x:i32, y:i32 }

var struct S @s = { 10, 20 }

fun struct S @get_struct() {
init:
  bid: b0
  allocations:

block b0:
  %b0:i0:struct S = load @s:struct S*
  ret %b0:i0:struct S
}
)ir";

    auto result = runFunction(&context, code, "get_struct", {});
    CHECK(result.size() == 1);
    VRegister *vReg = result[0].get();
    CHECK(vReg->isa<VRegisterDynamic>());
    auto structReg = vReg->cast<VRegisterDynamic>();
    CHECK(structReg->getSize() == 8);

    int32_t x = *reinterpret_cast<int32_t *>(structReg->getData());
    int32_t y = *reinterpret_cast<int32_t *>(
        static_cast<char *>(structReg->getData()) + 4);
    CHECK(x == 10);
    CHECK(y == 20);
  }

  SUBCASE("global struct test 2") {
    auto code = R"ir(
struct S : { x:i8, y:i32 }

var struct S @s = { 10, 20 }

fun struct S @get_struct() {
init:
  bid: b0
  allocations:

block b0:
  %b0:i0:struct S = load @s:struct S*
  ret %b0:i0:struct S
}
)ir";

    auto result = runFunction(&context, code, "get_struct", {});
    CHECK(result.size() == 1);
    VRegister *vReg = result[0].get();
    CHECK(vReg->isa<VRegisterDynamic>());
    auto structReg = vReg->cast<VRegisterDynamic>();
    CHECK(structReg->getSize() == 8);
    int8_t x = *reinterpret_cast<int8_t *>(structReg->getData());
    int32_t y = *reinterpret_cast<int32_t *>(
        static_cast<char *>(structReg->getData()) + 4);
    CHECK(x == 10);
    CHECK(y == 20);
  }

  SUBCASE("struct field access test") {
    auto code = R"ir(
struct S : { x:i32, y:i32 }
var struct S @s = { 10, 20 }
fun i32 @get_x() {
init:
  bid: b0
  allocations:

block b0:
  %b0:i0:i32* = getelementptr @s:struct S* offset 0:i64
  %b0:i1:i32 = load %b0:i0:i32*
  ret %b0:i1:i32
}
)ir";
    auto result = runFunction(&context, code, "get_x", {});
    CHECK(result.size() == 1);
    VRegister *vReg = result[0].get();
    CHECK(vReg->isa<VRegisterInt>());
    auto intReg = vReg->cast<VRegisterInt>();
    CHECK(intReg->getValue() == 10);
  }

  SUBCASE("complex struct") {
    auto code = R"ir(
struct S0 : { x:i8, y:i32 }
struct S1:  { a:i32, b:struct S0, c:i8 }

var struct S1 @s = { 100, { 10, 20 }, 200 }

fun struct S1 @get_struct() {
init:
  bid: b0
  allocations:

block b0:
  %b0:i0:struct S1 = load @s:struct S1*
  ret %b0:i0:struct S1
}
)ir";

    auto result = runFunction(&context, code, "get_struct", {});
    CHECK(result.size() == 1);
    VRegister *vReg = result[0].get();
    CHECK(vReg->isa<VRegisterDynamic>());
    auto structReg = vReg->cast<VRegisterDynamic>();
    CHECK(structReg->getSize() == 16);

    int32_t a = *reinterpret_cast<int32_t *>(structReg->getData());
    int8_t b_x = *reinterpret_cast<int8_t *>(
        static_cast<char *>(structReg->getData()) + 4);
    int32_t b_y = *reinterpret_cast<int32_t *>(
        static_cast<char *>(structReg->getData()) + 8);
    int8_t c = *reinterpret_cast<int8_t *>(
        static_cast<char *>(structReg->getData()) + 12);

    CHECK(a == 100);
    CHECK(b_x == 10);
    CHECK(b_y == 20);
    CHECK(c == static_cast<int8_t>(200));
  }

  SUBCASE("modify memory test") {
    auto code = R"ir(
fun unit @modify(u8*, u8) {
init:
  bid: b0
  allocations:
    %l0:u8:init

block b0:
  %b0:p0:u8*:ptr
  %b0:p1:u8:init

  %b0:i0:unit = store %b0:p1:u8 %l0:u8*
  j b1(0:i32)

block b1:
  %b1:p0:i32:i
  %b1:i0:i64 = typecast %b1:p0:i32 to i64
  %b1:i1:u8* = getelementptr %b0:p0:u8* offset %b1:i0:i64
  %b1:i2:u8 = load %b1:i1:u8*
  %b1:i3:i1 = cmp ne %b1:i2:u8 0:u8
  br %b1:i3:i1, b2(), b4()

block b2:
  %b2:i0:u8 = load %l0:u8*
  %b2:i1:u8 = add %b2:i0:u8 1:u8
  %b2:i2:unit = store %b2:i1:u8 %l0:u8*
  %b2:i3:unit = store %b2:i0:u8 %b1:i1:u8*
  j b3()

block b3:
  %b3:i0:i32 = add %b1:p0:i32 1:i32
  j b1(%b3:i0:i32)

block b4:
  ret unit:unit
}
)ir";

    std::vector<char> buffer(11);
    const char *data = "xxxxxxxxxx";
    std::copy(data, data + 11, buffer.data());
    auto dataPtr = std::make_unique<VRegisterInt>();
    dataPtr->setValue(buffer.data());
    auto valR = std::make_unique<VRegisterInt>();
    valR->setValue('a');
    runFunction(&context, code, "modify", {dataPtr.get(), valR.get()});

    CHECK(buffer[0] == 'a');
    CHECK(buffer[1] == 'b');
    CHECK(buffer[2] == 'c');
    CHECK(buffer[3] == 'd');
    CHECK(buffer[4] == 'e');
    CHECK(buffer[5] == 'f');
    CHECK(buffer[6] == 'g');
    CHECK(buffer[7] == 'h');
    CHECK(buffer[8] == 'i');
    CHECK(buffer[9] == 'j');
    CHECK(buffer[10] == '\0');
  }
}

} // namespace kecc::ir
