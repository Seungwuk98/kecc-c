#include "TestUtils.h"
#include "kecc/c/ParseAST.h"

namespace kecc::c {

TEST_CASE("Parse and dump AST") {

  std::string diags;
  llvm::raw_string_ostream diagOS(diags);

  ParseAST ast(R"c(
struct S {
  int a;
  int b;
};

int main(int) { 
  int x = 4 * 5;
  int y = (float)3;
  __int128 z;
  return 0; 
}
)c",
               "test.c", diagOS, true);
  auto result = ast.parse();
  CHECK(result.isError());

  STR_EQ(diags, R"(
test.c:10:3: error: unsupported type : __int128
   10 |   __int128 z;
      |   ^
)");
}

} // namespace kecc::c
