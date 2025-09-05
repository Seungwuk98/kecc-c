#ifndef KECC_LEXER_H
#define KECC_LEXER_H

#include "kecc/ir/Context.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"

namespace kecc {

using llvm::SMLoc;
using llvm::SMRange;
using llvm::StringRef;
using std::size_t;
using std::string;

class Token {
public:
  enum Kind {
#define TOKEN(Name) Tok_##Name,
#include "kecc/parser/Tokens.def"
  };

  Token(SMRange range, Kind kind, StringRef symbol, size_t row, size_t col,
        size_t index)
      : range(range), kind(kind), symbol(symbol), row(row), col(col),
        index(index) {}

  SMRange getRange() const { return range; }
  Kind getKind() const { return kind; }
  StringRef getSymbol() const { return symbol; }
  size_t getRow() const { return row; }
  size_t getCol() const { return col; }
  size_t getIndex() const { return index; }

  bool operator==(const Token &other) const {
    return kind == other.kind && symbol == other.symbol && row == other.row &&
           col == other.col && index == other.index;
  }

  template <Kind... K> bool is() const { return ((kind == K) || ...); }

  string toString() const;

  size_t getNumberFromId() const;

private:
  SMRange range;
  Kind kind;
  StringRef symbol;
  size_t row;
  size_t col;
  size_t index;
};

llvm::StringRef tokenKindToString(Token::Kind kind);

class Lexer {
public:
  Lexer(llvm::StringRef buffer, ir::IRContext *context)
      : buffer(buffer), context(context) {}

  enum class LexMode {
    General,
    RegisterId,
    Constant,
    Initializer,
  };

  Token *nextToken(LexMode mode = LexMode::General);
  Token *prevToken() const;
  Token *peekToken(size_t offset = 0, LexMode mode = LexMode::General);
  Token *tryLex(llvm::StringRef data, Token::Kind success);

  void rollback(Token *token);

private:
  static constexpr char CHAR_EOF = static_cast<char>(EOF);

  llvm::StringRef buffer;
  size_t cursor = 0;
  size_t col = 0;
  size_t row = 0;
  size_t savedCursor = 0;
  size_t savedCol = 0;
  size_t savedRow = 0;
  Token::Kind currentKind = Token::Tok_unknown;

  size_t tokenCursor = 0;

  void lexGeneral();
  void lexInitializer();
  void lexRegisterId();
  void lexConstant();
  bool lexNumberSequence();
  void lexNumber();
  void lexNumberValue();
  void lexIdentifier();
  void lexName();
  void lexIdentSequence(llvm::StringRef sequence, Token::Kind success,
                        Token::Kind failure = Token::Tok_unknown);
  void lexToTokenIndex(size_t index, LexMode mode);

  void lexLineComment();
  bool lexBlockComment();

  char advance();
  char peek() const;
  char prev() const;

  void capture();

  void skipWhitespace();

  Token *create();

  std::vector<Token *> tokens;
  ir::IRContext *context;
};

} // namespace kecc

#endif // KECC_LEXER_H
