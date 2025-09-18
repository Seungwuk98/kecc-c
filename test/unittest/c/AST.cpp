#include "TestUtils.h"
#include "kecc/c/ParseAST.h"

namespace kecc::c {

TEST_CASE("Parse and dump AST") {
  ParseAST ast(R"c(
struct S {
  int a;
  int b;
};

int main() { 
  int x = 4 * 5;
  int y = (float)3;
  return 0; 
}
)c",
               "test.c");
  ast.parse();
  ast.dump();
}

} // namespace kecc::c
