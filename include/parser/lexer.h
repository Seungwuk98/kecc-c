#ifndef KECC_LEXER_H
#define KECC_LEXER_H

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

  string toString() const;

private:
  SMRange range;
  Kind kind;
  StringRef symbol;
  size_t row;
  size_t col;
  size_t index;
};

} // namespace kecc

#endif // KECC_LEXER_H
