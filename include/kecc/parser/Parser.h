#ifndef KECC_PARSER_H
#define KECC_PARSER_H

#include "kecc/ir/Context.h"
#include "kecc/ir/IR.h"
#include "kecc/ir/IRAttributes.h"
#include "kecc/ir/IRBuilder.h"
#include "kecc/ir/IRInstructions.h"
#include "kecc/ir/Instruction.h"
#include "kecc/ir/Module.h"
#include "kecc/parser/Lexer.h"
#include "kecc/utils/Diag.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/SourceMgr.h"
#include <memory>

namespace kecc {

namespace detail {
struct ParserDetail;
}

class Parser {
public:
  Parser(Lexer &lexer, ir::IRContext *context)
      : lexer(lexer), diag(context->diag()), context(context),
        builder(context) {}

  std::unique_ptr<ir::Module> parseAndBuildModule();
  std::unique_ptr<ir::IR> parse();

private:
  friend struct detail::ParserDetail;
  struct RangeHelper {
    RangeHelper(Parser &parser, Token *startTok) : parser(parser) {
      startLoc = startTok->getRange().Start;
    }

    llvm::SMRange getRange() const {
      auto endLoc = parser.prevToken()->getRange().End;
      return llvm::SMRange(startLoc, endLoc);
    }

  private:
    llvm::SMLoc startLoc;
    Parser &parser;
  };

  std::pair<llvm::StringRef, ir::Type> parseField();

  ir::inst::StructDefinition parseStructDefinition(ir::IR *program,
                                                   Token *startToken);

  ir::inst::GlobalVariableDefinition
  parseGlobalVariableDefinition(ir::IR *program, Token *startToken);

  ir::InitializerAttr parseInitializerAttr(ir::IR *program);

  ir::Function *parseFunction(ir::IR *program, Token *startToken);
  void parseFunctionBody(ir::Function *function);
  ir::inst::LocalVariable parseAllocation(ir::Block *parentBlock);
  ir::Block *parseBlock(ir::Function *parentFunction);

  void registerRid(ir::Value value, ir::RegisterId rid);
  ir::Instruction parseInstruction(ir::Block *parentBlock);
  ir::Instruction parseInstructionInner(ir::Block *parentBlock,
                                        llvm::ArrayRef<ir::Type> types);

  ir::BlockExit parseBlockExit(Token *firstToken, ir::Block *parentBlock);
  ir::JumpArgState parseJumpArg(ir::Block *parentBlock);

  ir::Value parseOperand(ir::Block *parentBlock);
  ir::ConstantAttr parseConstant(ir::Block *parentBlock, Token *startToken);

  std::optional<std::tuple<ir::RegisterId, ir::Type, llvm::StringRef>>
  parseRegisterId(ir::Block *parentBlock);

  // parse type
  // type := pointer
  ir::Type parseType();

  // parse pointer
  // pointer := const '*' const?
  ir::Type parsePointerType();

  // parse const
  // const := 'const'? raw_type
  ir::Type parseConstType();

  // parse raw type
  ir::Type parseRawType();

  Token *nextToken() { return lexer.nextToken(); }
  Token *nextRidToken() { return lexer.nextToken(Lexer::LexMode::RegisterId); }
  Token *nextConstantToken() {
    return lexer.nextToken(Lexer::LexMode::Constant);
  }
  Token *nextInitializerToken() {
    return lexer.nextToken(Lexer::LexMode::Initializer);
  }
  Token *prevToken() const { return lexer.prevToken(); }
  Token *peekToken(size_t offset = 0) { return lexer.peekToken(offset); }
  Token *peekRidToken(size_t offset = 0) {
    return lexer.peekToken(offset, Lexer::LexMode::RegisterId);
  }
  Token *peekInitializerToken(size_t offset = 0) {
    return lexer.peekToken(offset, Lexer::LexMode::Initializer);
  }

  template <Token::Kind K> bool expect(Token *token) {
    assert(token && "Token cannot be null in expect.");
    if (token->is<K>())
      return false;
    report<ParserDiag::Diag::unexpected_token>(
        token->getRange(), token->toString(), tokenKindToString(K));
    return true;
  }

  template <Token::Kind K> bool consume() {
    auto *token = nextToken();
    return expect<K>(token);
  }

  template <Token::Kind... K> bool consumeStream() {
    return (consume<K>() || ...);
  }

  template <Token::Kind K> bool consumeIf() {
    if (peekToken()->is<K>()) {
      nextToken();
      return true;
    }
    return false;
  }

  void initRegisterMap() { registerMap.clear(); }

  struct ParserDiag {
    enum Diag {
#define DIAG(Name, ...) Name,
#include "kecc/parser/ParserDiag.def"
    };

    static llvm::SourceMgr::DiagKind getDiagKind(Diag diag);
    static const char *getDiagMessage(Diag diag);
  };

  template <ParserDiag::Diag D, typename... Args>
  void report(llvm::SMLoc loc, llvm::SMRange range, Args &&...args) {
    diag.report(loc, range, ParserDiag::getDiagKind(D),
                llvm::formatv(ParserDiag::getDiagMessage(D),
                              std::forward<Args>(args)...)
                    .str());
  }

  template <ParserDiag::Diag D, typename... Args>
  void report(llvm::SMLoc loc, Args &&...args) {
    diag.report(loc, ParserDiag::getDiagKind(D),
                llvm::formatv(ParserDiag::getDiagMessage(D),
                              std::forward<Args>(args)...)
                    .str());
  }

  template <ParserDiag::Diag D, typename... Args>
  void report(llvm::SMRange range, Args &&...args) {
    report<D>(range.Start, range, std::forward<Args>(args)...);
  }

  void reportError(llvm::SMRange range, llvm::StringRef message) {
    diag.report(range, llvm::SourceMgr::DK_Error, message);
  }

  void reportError(llvm::SMLoc loc, llvm::StringRef message) {
    diag.report(loc, llvm::SourceMgr::DK_Error, message);
  }

  Lexer &lexer;
  utils::DiagEngine &diag;
  ir::IRContext *context;
  ir::IRBuilder builder;
  llvm::DenseMap<ir::RegisterId, ir::Value> registerMap;
  llvm::DenseMap<ir::Value, ir::Value> replaceMap;
};

} // namespace kecc

#endif // KECC_PARSER_H
