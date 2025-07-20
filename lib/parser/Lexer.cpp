#include "kecc/parser/Lexer.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <cctype>

namespace kecc {

size_t Token::getNumberFromId() const {
  assert(
      (is<Token::Tok_instruction_id, Token::Tok_phi_id, Token::Tok_block_id,
          Token::Tok_percent_block_id, Token::Tok_percent_allocation_id>()) &&
      "Token is not a ID.");

  auto numberStr = symbol[0] == '%' ? symbol.substr(2)
                                    : symbol.substr(1); // skip % and l|p|b
  std::size_t numberValue;
  numberStr.getAsInteger(10, numberValue);
  return numberValue;
}

static const char *tokenName[] = {
#define TOKEN(Name) #Name,
#include "kecc/parser/Tokens.def"
};

std::string Token::toString() const {
  return llvm::formatv("{0}({1})", tokenKindToString(kind), symbol).str();
}

llvm::StringRef tokenKindToString(Token::Kind kind) {
  return tokenName[static_cast<size_t>(kind)];
}

Token *Lexer::nextToken(LexMode mode) {
  if (tokenCursor >= tokens.size()) {
    lexToTokenIndex(tokenCursor, mode);
  }

  if (tokenCursor >= tokens.size()) {
    return tokens.back(); // return EOF
  }
  return tokens[tokenCursor++];
}

Token *Lexer::prevToken() const {
  assert(tokenCursor > 0 && "Cannot go back before the first token.");
  return tokens[tokenCursor - 1];
}

Token *Lexer::peekToken(size_t offset, LexMode mode) {
  auto index = tokenCursor + offset;
  if (index >= tokens.size()) {
    lexToTokenIndex(index, mode);
  }

  if (index >= tokens.size()) {
    return tokens.back(); // return EOF
  }

  return tokens[index];
}

Token *Lexer::tryLex(llvm::StringRef data, Token::Kind success) {
  assert(tokenCursor == tokens.size() &&
         "Cannot try lex when there are already tokens lexed. Use rollback() "
         "first.");
  skipWhitespace();
  capture();
  lexIdentSequence(data, success, Token::Tok_unknown);
  auto token = create();
  tokens.emplace_back(token);
  tokenCursor++;
  return token;
}

void Lexer::rollback(Token *token) {
  assert(token != nullptr && "Cannot rollback to a null token.");
  auto index = token->getIndex();
  tokenCursor = index;

  while (tokens.size() > index) {
    tokens.pop_back();
  }

  auto locPair =
      context->getSourceMgr().getLineAndColumn(token->getRange().Start);
  row = locPair.first - 1;
  col = locPair.second - 1;
  cursor = token->getRange().Start.getPointer() - buffer.data();
}

void Lexer::lexToTokenIndex(size_t index, LexMode mode) {
  while (tokens.size() <= index) {
    if (!tokens.empty() && tokens.back()->is<Token::Tok_EOF>())
      return;

    mode == LexMode::General      ? lexGeneral()
    : mode == LexMode::RegisterId ? lexRegisterId()
    : mode == LexMode::Constant   ? lexConstant()
                                  : lexInitializer();
    tokens.emplace_back(create());
  }
}

#define LEX_COMMENT(LexFunction)                                               \
  case '/':                                                                    \
    if (peek() == '/') {                                                       \
      while (peek() != '\n')                                                   \
        advance();                                                             \
      advance();                                                               \
      LexFunction();                                                           \
      break;                                                                   \
    }                                                                          \
    currentKind = Token::Tok_unknown;                                          \
    break;

void Lexer::lexGeneral() {
  skipWhitespace();
  capture();

  char c = advance();
  switch (c) {
  case '(':
    currentKind = Token::Tok_lparen;
    break;
  case ')':
    currentKind = Token::Tok_rparen;
    break;
  case '{':
    currentKind = Token::Tok_lbrace;
    break;
  case '}':
    currentKind = Token::Tok_rbrace;
    break;
  case '[':
    currentKind = Token::Tok_lbracket;
    break;
  case ']':
    currentKind = Token::Tok_rbracket;
    break;
  case ',':
    currentKind = Token::Tok_comma;
    break;
  case '=':
    currentKind = Token::Tok_equal;
    break;
  case ':':
    currentKind = Token::Tok_colon;
    break;
  case '-':
    currentKind = Token::Tok_minus;
    break;
  case '+':
    currentKind = Token::Tok_plus;
    break;
  case '*':
    currentKind = Token::Tok_asterisk;
    break;
  case CHAR_EOF:
    currentKind = Token::Tok_EOF;
    break;

    LEX_COMMENT(lexGeneral);

  default: {
    if (std::isalpha(c) || c == '_' || c == '%' || c == '@') {
      lexIdentifier();
    } else if (std::isdigit(c)) {
      lexNumberValue();
    } else {
      currentKind = Token::Tok_unknown;
    }
  }
  }
}

void Lexer::lexRegisterId() {
  skipWhitespace();
  capture();

  char c = advance();
  switch (c) {
  case '%': {
    c = advance();
    if (c == 'b') {
      currentKind = Token::Tok_percent_block_id;
    } else if (c == 'l') {
      currentKind = Token::Tok_percent_allocation_id;
    } else if (c == 't') {
      currentKind = Token::Tok_identifier;
    } else {
      currentKind = Token::Tok_unknown;
    }
    if (!lexNumberSequence())
      currentKind = Token::Tok_unknown;
    break;
  }
  case 'i':
    currentKind =
        lexNumberSequence() ? Token::Tok_instruction_id : Token::Tok_identifier;
    break;

  case 'p':
    currentKind =
        lexNumberSequence() ? Token::Tok_phi_id : Token::Tok_identifier;
    break;

  case 'b':
    currentKind =
        lexNumberSequence() ? Token::Tok_block_id : Token::Tok_identifier;
    break;

  case ':':
    currentKind = Token::Tok_colon;
    break;

    LEX_COMMENT(lexRegisterId);
  default:
    if (!std::isalpha(c) && c != '_') {
      currentKind = Token::Tok_unknown;
    } else {
      lexName();
      currentKind = Token::Tok_identifier;
    }
  }
}

void Lexer::lexConstant() {
  skipWhitespace();
  capture();

  char c = advance();
  switch (c) {
  case '+':
    currentKind = Token::Tok_plus;
    break;
  case '-':
    currentKind = Token::Tok_minus;
    break;
  case '@': {
    auto savedCursor = cursor;
    lexName();
    currentKind =
        savedCursor == cursor ? Token::Tok_unknown : Token::Tok_global_variable;
    break;
  }

    LEX_COMMENT(lexConstant);

  default:
    if (std::isdigit(c)) {
      lexNumberValue();
    } else if (std::isalpha(c)) {
      lexName();
      llvm::StringRef symbol = buffer.slice(savedCursor, cursor);
      currentKind = llvm::StringSwitch<Token::Kind>(symbol)
                        .Case("undef", Token::Tok_undef)
                        .Case("unit", Token::Tok_unit)
                        .Default(Token::Tok_unknown);
    } else {
      currentKind = Token::Tok_unknown;
    }
  }
}

void Lexer::lexInitializer() {
  skipWhitespace();
  capture();

  char c = advance();
  switch (c) {
  case '+':
    currentKind = Token::Tok_plus;
    break;
  case '-':
    currentKind = Token::Tok_minus;
    break;
  case '(':
    currentKind = Token::Tok_lparen;
    break;
  case ')':
    currentKind = Token::Tok_rparen;
    break;
  case '{':
    currentKind = Token::Tok_lbrace;
    break;
  case '}':
    currentKind = Token::Tok_rbrace;
    break;
  case ',':
    currentKind = Token::Tok_comma;
    break;

    LEX_COMMENT(lexInitializer);
  case '0': {
    char p = peek();
    if (p == 'x' || p == 'X') {
      // 0x[0-9a-fA-F]+[lL]?
      advance();

      auto savedCursor = cursor;
      lexNumber();
      if (savedCursor == cursor) {
        currentKind = Token::Tok_unknown;
        break;
      }

      savedCursor = cursor;
      lexName();
      if (savedCursor != cursor) {
        auto suffix = buffer.slice(savedCursor, cursor);
        currentKind = (suffix == "l" || suffix == "L") ? Token::Tok_ast_integer
                                                       : Token::Tok_unknown;
      }
      break;
    }
  }
  default: {
    if (std::isdigit(c)) {
      lexNumber();
      char p = peek();
      if (p == '.') {
        // 0.[0-9]*[fF]?
        advance();
        lexNumber();

        auto savedCursor = cursor;
        lexName();
        currentKind = Token::Tok_ast_float;
        if (savedCursor != cursor) {
          auto suffix = buffer.slice(savedCursor, cursor);
          currentKind = (suffix == "f" || suffix == "F") ? Token::Tok_ast_float
                                                         : Token::Tok_unknown;
        }
      } else {
        // [0-9]+[lL]
        auto savedCursor = cursor;
        lexName();
        currentKind = Token::Tok_ast_integer;
        if (savedCursor != cursor) {
          auto suffix = buffer.slice(savedCursor, cursor);
          currentKind = suffix == "l" || suffix == "L" ? Token::Tok_ast_integer
                                                       : Token::Tok_unknown;
        }
      }
    } else {
      currentKind = Token::Tok_unknown;
    }
  }
  }
}

bool Lexer::lexNumberSequence() {
  auto savedCursor = cursor;
  lexName();
  llvm::StringRef symbol = buffer.slice(savedCursor, cursor);
  if (symbol.empty())
    return false;
  if (llvm::any_of(symbol, [](char c) { return !std::isdigit(c); })) {
    return false;
  }
  return true;
}

void Lexer::lexNumber() {
  char c = peek();
  while (c != CHAR_EOF && std::isdigit(c)) {
    advance();
    c = peek();
  }
}

void Lexer::lexNumberValue() {
  lexNumber();
  char c = peek();
  if (c == '.') {
    advance();
    lexNumber();
    currentKind = Token::Tok_float;
  } else {
    currentKind = Token::Tok_integer;
  }
}

void Lexer::lexIdentifier() {
  if (prev() == '%') {
    char next = advance();
    switch (next) {
    case 't': {
      currentKind =
          !lexNumberSequence() ? Token::Tok_unknown : Token::Tok_identifier;
      break;
    }
    case 'a':
      lexIdentSequence("non", Token::Tok_anonymous);
      break;
    default:
      currentKind = Token::Tok_unknown;
    }
  } else if (prev() == '@') {
    char p = peek();
    if (!std::isalpha(p) && p != '_') {
      currentKind = Token::Tok_unknown;
      return;
    }
    advance();
    lexName();
    currentKind = Token::Tok_global_variable;
  } else {
    lexName();
    llvm::StringRef symbol = buffer.slice(savedCursor, cursor);

    currentKind = llvm::StringSwitch<Token::Kind>(symbol)
#define KEYWORD(Name) .Case(#Name, Token::Kind::Tok_##Name)
#define TYPE(Name) .Case(#Name, Token::Kind::Tok_##Name)
#include "kecc/parser/Tokens.def"
                      .Default(Token::Tok_identifier);
  }
}

void Lexer::lexName() {
  char c = peek();
  while (c != CHAR_EOF && (std::isalnum(c) || c == '_')) {
    advance();
    c = peek();
  }
}

void Lexer::lexIdentSequence(llvm::StringRef sequence, Token::Kind success,
                             Token::Kind failure) {
  auto startCursor = cursor;
  lexName();

  llvm::StringRef symbol = buffer.slice(startCursor, cursor);
  currentKind = symbol == sequence ? success : failure;
}

char Lexer::advance() {
  if (cursor >= buffer.size()) {
    return CHAR_EOF;
  }

  char c = buffer[cursor++];

  if (c == '\n') {
    row++;
    col = 0;
  } else {
    col++;
  }

  return c;
}

char Lexer::peek() const {
  if (cursor >= buffer.size()) {
    return CHAR_EOF;
  }

  return buffer[cursor];
}

char Lexer::prev() const {
  assert(cursor > 0 && "Cannot peek before the start of the buffer.");
  return buffer[cursor - 1];
}

void Lexer::capture() {
  savedCursor = cursor;
  savedCol = col;
  savedRow = row;
}

void Lexer::skipWhitespace() {
  char c = peek();
  while (std::isspace(c)) {
    advance();
    c = peek();
  }
}

Token *Lexer::create() {
  SMLoc start = SMLoc::getFromPointer(buffer.data() + savedCursor);
  SMLoc end = SMLoc::getFromPointer(buffer.data() + cursor);
  SMRange range(start, end);

  StringRef symbol = buffer.slice(savedCursor, cursor);

  Token *token = new (context->allocate(sizeof(Token)))
      Token(range, currentKind, symbol, savedRow, savedCol, tokens.size());

  return token;
}

} // namespace kecc
