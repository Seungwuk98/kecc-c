#include "TestUtils.h"
#include "kecc/ir/Instruction.h"
#include "kecc/ir/Module.h"
#include "kecc/parser/Parser.h"
#include "llvm/Support/MemoryBuffer.h"

namespace kecc {

TEST_CASE("Lexer Tests") {

  ir::IRContext context;
  SUBCASE("Lexing program") {
    auto code = R"(
fun i32 @main() {
init:
  bid: b0
  allocations:
    %l0:i32:x

block b0:
  %b0:i0:i32 = load %l0:i32*
  ret %b0:i0:i32
}
)";

    ir::IRContext context;
    auto buffer = llvm::MemoryBuffer::getMemBuffer(code, "test_ir");
    auto index = context.getSourceMgr().AddNewSourceBuffer(std::move(buffer),
                                                           llvm::SMLoc());
    auto bufferRef = context.getSourceMgr().getMemoryBuffer(index);

    Lexer lexer(bufferRef->getBuffer(), &context);
    auto fun0 = lexer.nextToken();
    CHECK(fun0->getKind() == Token::Tok_fun);
    auto i320 = lexer.nextToken();
    CHECK(i320->getKind() == Token::Tok_i32);
    auto global_main = lexer.nextToken();
    CHECK(global_main->getKind() == Token::Tok_global_variable);
    CHECK(global_main->getSymbol() == "@main");
    auto lparen0 = lexer.nextToken();
    CHECK(lparen0->getKind() == Token::Tok_lparen);
    auto rparen0 = lexer.nextToken();
    CHECK(rparen0->getKind() == Token::Tok_rparen);
    auto lbrace0 = lexer.nextToken();
    CHECK(lbrace0->getKind() == Token::Tok_lbrace);
    auto init = lexer.nextToken();
    CHECK(init->getKind() == Token::Tok_init);
    auto colon0 = lexer.nextToken();
    CHECK(colon0->getKind() == Token::Tok_colon);
    auto bid = lexer.nextToken();
    CHECK(bid->getKind() == Token::Tok_bid);
    auto colon1 = lexer.nextToken();
    CHECK(colon1->getKind() == Token::Tok_colon);
    auto bid0_0 = lexer.nextToken(Lexer::LexMode::RegisterId);
    CHECK(bid0_0->getKind() == Token::Tok_block_id);
    CHECK(bid0_0->getSymbol() == "b0");
    auto allocations = lexer.nextToken();
    CHECK(allocations->getKind() == Token::Tok_allocations);
    auto colon2 = lexer.nextToken();
    CHECK(colon2->getKind() == Token::Tok_colon);
    auto p_l0_0 = lexer.nextToken(Lexer::LexMode::RegisterId);
    CHECK(p_l0_0->getKind() == Token::Tok_percent_allocation_id);
    CHECK(p_l0_0->getSymbol() == "%l0");
    auto colon3 = lexer.nextToken();
    CHECK(colon3->getKind() == Token::Tok_colon);
    auto i32_1 = lexer.nextToken();
    CHECK(i32_1->getKind() == Token::Tok_i32);
    auto colon4 = lexer.nextToken();
    CHECK(colon4->getKind() == Token::Tok_colon);
    auto x = lexer.nextToken(Lexer::LexMode::RegisterId);
    CHECK(x->getKind() == Token::Tok_identifier);
    CHECK(x->getSymbol() == "x");
    auto block0 = lexer.nextToken();
    CHECK(block0->getKind() == Token::Tok_block);
    auto bid0_1 = lexer.nextToken(Lexer::LexMode::RegisterId);
    CHECK(bid0_1->getKind() == Token::Tok_block_id);
    CHECK(bid0_1->getSymbol() == "b0");
    auto colon5 = lexer.nextToken();
    CHECK(colon5->getKind() == Token::Tok_colon);
    auto p_b0_0 = lexer.nextToken(Lexer::LexMode::RegisterId);
    CHECK(p_b0_0->getKind() == Token::Tok_percent_block_id);
    CHECK(p_b0_0->getSymbol() == "%b0");
    auto colon6 = lexer.nextToken();
    CHECK(colon6->getKind() == Token::Tok_colon);
    auto i0 = lexer.nextToken(Lexer::LexMode::RegisterId);
    CHECK(i0->getKind() == Token::Tok_instruction_id);
    CHECK(i0->getSymbol() == "i0");
    auto colon7 = lexer.nextToken();
    CHECK(colon7->getKind() == Token::Tok_colon);
    auto i32_2 = lexer.nextToken();
    CHECK(i32_2->getKind() == Token::Tok_i32);
    auto equal = lexer.nextToken();
    CHECK(equal->getKind() == Token::Tok_equal);
    auto load = lexer.nextToken();
    CHECK(load->getKind() == Token::Tok_identifier);
    CHECK(load->getSymbol() == "load");
    auto p_l0_1 = lexer.nextToken(Lexer::LexMode::RegisterId);
    CHECK(p_l0_1->getKind() == Token::Tok_percent_allocation_id);
    CHECK(p_l0_1->getSymbol() == "%l0");
    auto colon8 = lexer.nextToken();
    CHECK(colon8->getKind() == Token::Tok_colon);
    auto i32_3 = lexer.nextToken();
    CHECK(i32_3->getKind() == Token::Tok_i32);
    auto star = lexer.nextToken();
    CHECK(star->getKind() == Token::Tok_asterisk);
    auto ret = lexer.nextToken();
    CHECK(ret->getKind() == Token::Tok_ret);
    auto p_b0_1 = lexer.nextToken(Lexer::LexMode::RegisterId);
    CHECK(p_b0_1->getKind() == Token::Tok_percent_block_id);
    CHECK(p_b0_1->getSymbol().str() == "%b0");
    auto colon9 = lexer.nextToken();
    CHECK(colon9->getKind() == Token::Tok_colon);
    auto i0_1 = lexer.nextToken(Lexer::LexMode::RegisterId);
    CHECK(i0_1->getKind() == Token::Tok_instruction_id);
    CHECK(i0_1->getSymbol() == "i0");
    auto colon10 = lexer.nextToken();
    CHECK(colon10->getKind() == Token::Tok_colon);
    auto i32_4 = lexer.nextToken();
    CHECK(i32_4->getKind() == Token::Tok_i32);
    auto rbrace0 = lexer.nextToken();
    CHECK(rbrace0->getKind() == Token::Tok_rbrace);
  }
}

std::unique_ptr<ir::Module> testParseIR(ir::IRContext *context,
                                        llvm::StringRef code) {
  auto buffer = llvm::MemoryBuffer::getMemBuffer(code, "test_ir");
  auto index = context->getSourceMgr().AddNewSourceBuffer(std::move(buffer),
                                                          llvm::SMLoc());
  auto bufferRef = context->getSourceMgr().getMemoryBuffer(index);

  Lexer lexer(bufferRef->getBuffer(), context);
  Parser parser(lexer, context);

  auto module = parser.parseAndBuildModule();
  if (context->diag().hasError()) {
    FAIL("Parsing failed with {} errors.", context->diag().getErrorCount());
    return nullptr;
  }

  return module;
}

TEST_CASE("Parser Tests") {
  SUBCASE("Parsing simple program") {
    ir::IRContext context;
    auto module = testParseIR(&context, R"(
fun i32 @main() {
init:
  bid: b0
  allocations:
    %l0:i32:x

block b0:
  %b0:i0:i32 = load %l0:i32*
  ret %b0:i0:i32
}
)");

    auto *ir = module->getIR();
    std::string result;
    llvm::raw_string_ostream os(result);
    ir::IRPrintContext printContext(os, ir::IRPrintContext::Default);
    ir->print(printContext);

    STR_EQ(result, R"(
fun i32 @main () {
init:
  bid: b0
  allocations:
    %l0:i32:x

block b0:
  %b0:i0:i32 = load %l0:i32*
  ret %b0:i0:i32
})");
  }
}

} // namespace kecc
