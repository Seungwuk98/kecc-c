#include "kecc/parser/Parser.h"
#include "kecc/ir/Block.h"
#include "kecc/ir/Context.h"
#include "kecc/ir/IRAttributes.h"
#include "kecc/ir/IRBuilder.h"
#include "kecc/ir/IRInstructions.h"
#include "kecc/ir/IRTypes.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/SourceMgr.h"

namespace kecc {

std::unique_ptr<ir::Module> Parser::parseAndBuildModule() {
  auto ir = parse();
  if (diag.hasError())
    return nullptr;

  auto module = ir::Module::create(std::move(ir));
  for (const auto &[from, to] : replaceMap) {
    auto unresolved =
        from.getInstruction()->getDefiningInst<ir::inst::Unresolved>();
    assert(unresolved && "Expected unresolved instruction for replacement");

    module->replaceInst(unresolved.getStorage(), to, true);
  }

  return module;
}

std::unique_ptr<ir::IR> Parser::parse() {
  auto program = std::make_unique<ir::IR>(context);

  auto nextToken = lexer.nextToken();

  while (!nextToken->is<Token::Tok_EOF>()) {
    if (nextToken->is<Token::Tok_struct>()) {
      parseStructDefinition(program.get(), nextToken);
    } else if (nextToken->is<Token::Tok_var>()) {
      parseGlobalVariableDefinition(program.get(), nextToken);
    } else if (nextToken->is<Token::Tok_fun>()) {
      parseFunction(program.get(), nextToken);
    } else {
      reportError(
          nextToken->getRange(),
          "Cannot parse token for (struct | global variable | function)");
      return nullptr;
    }
    if (diag.hasError())
      return nullptr;

    nextToken = lexer.nextToken();
  }

  return program;
}

std::pair<llvm::StringRef, ir::Type> Parser::parseField() {
  // identifier ':' type

  llvm::StringRef fieldName;
  auto *token = nextToken();
  if (token->is<Token::Tok_identifier>()) {
    fieldName = token->getSymbol();
  } else if (expect<Token::Tok_anonymous>(token))
    return {"", nullptr};

  if (consume<Token::Tok_colon>())
    return {"", nullptr};

  auto type = parseType();
  if (diag.hasError())
    return {"", nullptr};

  return {fieldName, type};
}

ir::inst::StructDefinition Parser::parseStructDefinition(ir::IR *program,
                                                         Token *startToken) {
  // 'struct' identifier ':' 'opaque'
  // 'struct' identifier ':' '{' identifier ':' type (',' identifier ':' type) *
  // '}'

  RangeHelper rh(*this, startToken);
  ir::IRBuilder::InsertionGuard guard(builder, program->getStructBlock());

  Token *token = nextToken();
  if (expect<Token::Tok_identifier>(token))
    return nullptr;

  llvm::StringRef name = token->getSymbol();

  if (consume<Token::Tok_colon>())
    return nullptr;

  if (consumeIf<Token::Tok_opaque>()) {
    return builder.create<ir::inst::StructDefinition>(
        rh.getRange(), llvm::ArrayRef<std::pair<llvm::StringRef, ir::Type>>(),
        name);
  }

  if (consume<Token::Tok_lbrace>())
    return nullptr;

  llvm::SmallVector<std::pair<llvm::StringRef, ir::Type>> fields;

  auto [fieldName, type] = parseField();
  if (diag.hasError())
    return nullptr;

  fields.emplace_back(fieldName, type);

  while (consumeIf<Token::Tok_comma>()) {
    auto [nextFieldName, nextType] = parseField();
    if (diag.hasError())
      return nullptr;

    fields.emplace_back(nextFieldName, nextType);
  }

  if (consume<Token::Tok_rbrace>())
    return nullptr;

  return builder.create<ir::inst::StructDefinition>(rh.getRange(), fields,
                                                    name);
}

ir::inst::GlobalVariableDefinition
Parser::parseGlobalVariableDefinition(ir::IR *program, Token *startToken) {
  // 'var' type global_variable ('=' initializer)?

  RangeHelper rh(*this, startToken);
  ir::IRBuilder::InsertionGuard guard(builder, program->getGlobalBlock());

  auto type = parseType();
  if (diag.hasError())
    return nullptr;

  auto *nameToken = nextToken();
  if (expect<Token::Tok_global_variable>(nameToken))
    return nullptr;

  // remove '@'
  llvm::StringRef name = nameToken->getSymbol().substr(1);

  if (consumeIf<Token::Tok_equal>()) {
    auto initializer = parseInitializerAttr(program);
    if (diag.hasError())
      return nullptr;

    return builder.create<ir::inst::GlobalVariableDefinition>(
        rh.getRange(), type, name, initializer);
  }

  return builder.create<ir::inst::GlobalVariableDefinition>(rh.getRange(), type,
                                                            name);
}

ir::Function *Parser::parseFunction(ir::IR *program, Token *startToken) {
  // 'fun' (type (',' type)*) global_variable '(' type (',' type)*
  // ')' func_body?
  RangeHelper rh(*this, startToken);

  llvm::SmallVector<ir::Type> retTypes;
  auto retType = parseType();
  if (diag.hasError())
    return nullptr;

  retTypes.emplace_back(retType);
  while (consumeIf<Token::Tok_comma>()) {
    retType = parseType();
    if (diag.hasError())
      return nullptr;

    retTypes.emplace_back(retType);
  }

  auto *nameToken = nextToken();
  if (expect<Token::Tok_global_variable>(nameToken))
    return nullptr;

  // remove '@'
  llvm::StringRef name = nameToken->getSymbol().substr(1);
  if (auto prevFunc = program->getFunction(name)) {
    diag.report(nameToken->getRange(), llvm::SourceMgr::DK_Error,
                "Function is already defined");
    diag.report(prevFunc->getRange(), llvm::SourceMgr::DK_Note,
                "Previous defined here");
    return nullptr;
  }

  if (consume<Token::Tok_lparen>())
    return nullptr;

  llvm::SmallVector<ir::Type> argTypes;

  if (!consumeIf<Token::Tok_rparen>()) {
    auto argType = parseType();
    if (diag.hasError())
      return nullptr;

    argTypes.emplace_back(argType);
    while (consumeIf<Token::Tok_comma>()) {
      auto nextArgType = parseType();
      if (diag.hasError())
        return nullptr;

      argTypes.emplace_back(nextArgType);
    }
    if (consume<Token::Tok_rparen>())
      return nullptr;
  }

  auto funcType = ir::FunctionT::get(context, retTypes, argTypes);

  ir::Function *func =
      new ir::Function(rh.getRange(), name, funcType, program, context);

  auto peekToken = lexer.peekToken();
  if (peekToken->is<Token::Tok_lbrace>()) {
    parseFunctionBody(func);
    if (diag.hasError()) {
      delete func;
      return nullptr;
    }
  } else {
    lexer.rollback(peekToken);
  }

  program->addFunction(func);

  return func;
}

void Parser::parseFunctionBody(ir::Function *function) {
  // '{'
  //    'init' ':'
  //       'bid' ':' bid
  //    'allocations':
  //       alloc_instruction*
  //
  //    block+
  // '}'

  initRegisterMap();
  auto lbrace = nextToken();
  RangeHelper rh(*this, lbrace);
  if (expect<Token::Tok_lbrace>(lbrace))
    return;
  if (consumeStream<Token::Tok_init, Token::Tok_colon, Token::Tok_bid,
                    Token::Tok_colon>())
    return;

  auto blockIdToken = nextRidToken();
  if (expect<Token::Tok_block_id>(blockIdToken))
    return;

  auto blockId = blockIdToken->getNumberFromId();
  function->setEntryBlock(blockId);

  if (consumeStream<Token::Tok_allocations, Token::Tok_colon>())
    return;

  auto ridTok = peekRidToken();
  {
    ir::IRBuilder::InsertionGuard guard(builder,
                                        function->getAllocationBlock());
    while (true) {
      if (ridTok->is<Token::Tok_percent_allocation_id>()) {
        auto ridOpt = parseRegisterId(function->getAllocationBlock());
        if (diag.hasError())
          return;
        auto [rid, type, name] = *ridOpt;
        auto allocInst = builder.create<ir::inst::LocalVariable>(
            rh.getRange(), ir::PointerT::get(context, type));
        allocInst.setValueName(name);

        auto [_, inserted] =
            registerMap.try_emplace(rid, allocInst.getResult());
        if (!inserted) {
          report<ParserDiag::duplicated_register_id>(allocInst.getRange(),
                                                     rid.toString());
          return;
        }
        ridTok = peekRidToken();
      } else {
        lexer.rollback(ridTok);
        break;
      }
    }
  }

  auto peekTok = peekToken();
  if (!peekTok->is<Token::Tok_block>()) {
    reportError(peekTok->getRange(),
                "Expected a block definition after allocations");
    return;
  }
  while (peekTok->is<Token::Tok_block>()) {
    parseBlock(function);
    if (diag.hasError())
      return;
    peekTok = peekToken();
  }

  consume<Token::Tok_rbrace>();
}

ir::Block *Parser::parseBlock(ir::Function *parentFunction) {
  Token *token = nextToken();
  RangeHelper rh(*this, token);
  if (expect<Token::Tok_block>(token))
    return nullptr;

  auto bidTok = nextRidToken();
  if (expect<Token::Tok_block_id>(bidTok))
    return nullptr;

  if (consume<Token::Tok_colon>())
    return nullptr;

  auto blockId = bidTok->getNumberFromId();
  ir::Block *block = parentFunction->getBlockById(blockId);
  /// 'addBlock' must be called whether the block already exists or not,
  /// because 'addBlock' rearranges the block list order in the function.
  block = parentFunction->addBlock(blockId);

  ir::IRBuilder::InsertionGuard guard(builder, block);

  while (true) {
    auto inst = parseInstruction(block);
    if (diag.hasError())
      return nullptr;
    if (inst.isa<ir::BlockExit>())
      break;
  }

  return block;
}

void Parser::registerRid(ir::Value value, ir::RegisterId rid) {
  auto [it, inserted] = registerMap.try_emplace(rid, value);

  if (!inserted) {
    auto exInst = it->second.getInstruction();
    if (!exInst->getDefiningInst<ir::inst::Unresolved>()) {
      report<ParserDiag::duplicated_register_id>(rid.getRange(),
                                                 rid.toString());
      report<ParserDiag::declaration_info>(exInst->getRange());
      return;
    }
    replaceMap.try_emplace(exInst->getDefiningInst<ir::inst::Unresolved>(),
                           value);
  }
}

ir::Instruction Parser::parseInstruction(ir::Block *parentBlock) {
  // block exits
  // (registerId ('|' registerId)*) ('=' instruction-inner)?

  auto *token = nextToken();
  auto blockExit = parseBlockExit(token, parentBlock);
  if (diag.hasError())
    return nullptr;

  if (blockExit)
    return blockExit;

  // Token is already lexed to general mode, so we can rollback to the start
  lexer.rollback(token);

  auto regIdOpt = parseRegisterId(parentBlock);
  if (diag.hasError())
    return nullptr;

  auto [registerId, type, name] = *regIdOpt;
  if (registerId.isArg()) {
    auto phi = builder.create<ir::Phi>(registerId.getRange(), type);
    phi.setValueName(name);
    registerRid(phi, registerId);
    if (diag.hasError())
      return nullptr;
    return phi;
  }

  if (registerId.isAlloc()) {
    auto alloc = builder.create<ir::inst::LocalVariable>(
        registerId.getRange(), ir::PointerT::get(context, type));
    registerRid(alloc, registerId);
    if (diag.hasError())
      return nullptr;
    return alloc;
  }

  llvm::SmallVector<ir::RegisterId> registers{registerId};
  llvm::SmallVector<ir::Type> types{type};
  llvm::SmallVector<llvm::StringRef> names{name};

  while (consumeIf<Token::Tok_comma>()) {
    auto nextRegIdOpt = parseRegisterId(parentBlock);
    if (diag.hasError())
      return nullptr;

    auto [nextRegisterId, nextType, nextName] = *nextRegIdOpt;
    if (nextRegisterId.isArg()) {
      reportError(nextRegisterId.getRange(),
                  "Cannot use argument register in instruction definition");
      return nullptr;
    }

    if (nextRegisterId.isAlloc()) {
      reportError(nextRegisterId.getRange(),
                  "Cannot use allocation register in instruction definition");
      return nullptr;
    }

    registers.emplace_back(nextRegisterId);
    types.emplace_back(nextType);
    names.emplace_back(nextName);
  }

  // instruction

  if (consume<Token::Tok_equal>())
    return nullptr;

  auto inst = parseInstructionInner(parentBlock, types /* for call op */);
  if (diag.hasError())
    return nullptr;

  if (inst.getStorage()->getResultSize() != registers.size()) {
    report<ParserDiag::unmatched_result_size>(
        inst.getRange(), inst.getStorage()->getResultSize(), registers.size());
    return nullptr;
  }

  for (size_t idx = 0; idx < registers.size(); ++idx) {
    if (inst.getStorage()->getResult(idx).getType().constCanonicalize() !=
        types[idx].constCanonicalize()) {
      report<ParserDiag::unmatched_result_type>(
          registers[idx].getRange(),
          inst.getStorage()->getResult(idx).getType().toString(),
          types[idx].toString());
      return nullptr;
    }

    if (!names[idx].empty())
      inst.getStorage()->getResult(idx).getImpl()->setValueName(names[idx]);

    registerRid(inst.getStorage()->getResult(idx), registers[idx]);
    if (diag.hasError())
      return nullptr;
  }

  return inst;
}

static llvm::StringRef handleAsIdentifier(Token *token) {
  if (token->is<Token::Tok_phi_id, Token::Tok_instruction_id,
                Token::Tok_identifier>()) {
    return token->getSymbol();
  }
  return "";
}

std::optional<std::tuple<ir::RegisterId, ir::Type, llvm::StringRef>>
Parser::parseRegisterId(ir::Block *parentBlock) {
  // '%l' number ':' type (':' 'name')?
  // '%b' number ':' ('p' | 'i') number ':' type (':' 'name')?

  auto *token = nextRidToken();
  RangeHelper rh(*this, token);
  if (token->is<Token::Tok_percent_block_id>()) {
    auto bidTok = token;
    if (consume<Token::Tok_colon>())
      return {};

    token = nextRidToken();
    if (!token->is<Token::Tok_phi_id, Token::Tok_instruction_id>()) {
      reportError(token->getRange(),
                  "Expected a register id (phi or instruction) after block id");
      return {};
    }

    auto instIdTok = token;

    if (consume<Token::Tok_colon>())
      return {};

    auto type = parseType();
    if (diag.hasError())
      return {};

    llvm::StringRef name;
    if (consumeIf<Token::Tok_colon>()) {
      auto nameTok = nextRidToken();
      name = handleAsIdentifier(nameTok);
      if (name.empty()) {
        reportError(nameTok->getRange(), "Expected a valid identifier");
        return {};
      }
    } else {
      lexer.rollback(peekToken());
    }

    auto blockId = bidTok->getNumberFromId();
    auto regKind = instIdTok->is<Token::Tok_phi_id>()
                       ? ir::RegisterId::Kind::Arg
                       : ir::RegisterId::Kind::Temp;
    auto regId = instIdTok->getNumberFromId();

    auto result = instIdTok->is<Token::Tok_phi_id>()
                      ? ir::RegisterId::arg(rh.getRange(), blockId, regId)
                      : ir::RegisterId::temp(rh.getRange(), blockId, regId);

    return std::tuple{result, type, name};
  } else if (token->is<Token::Kind::Tok_percent_allocation_id>()) {
    auto allocIdTok = token;
    if (consume<Token::Tok_colon>())
      return {};

    auto type = parseType();
    if (diag.hasError())
      return {};

    llvm::StringRef name;
    if (consumeIf<Token::Tok_colon>()) {
      auto nameTok = nextRidToken();
      name = handleAsIdentifier(nameTok);
      if (name.empty()) {
        reportError(nameTok->getRange(), "Expected a valid identifier");
        return {};
      }
    } else {
      lexer.rollback(peekToken());
    }

    auto blockId = parentBlock->getId();
    auto allocId = allocIdTok->getNumberFromId();
    auto result = ir::RegisterId::alloc(rh.getRange(), allocId);
    return std::tuple{result, type, name};
  }

  reportError(token->getRange(), "Expected a valid register id");
  assert(false);
  return {};
}

namespace detail {
struct ParserDetail {
  static ir::Instruction parseNop(Parser &parser, Token *startTok,
                                  ir::Block *parentBlock,
                                  llvm::ArrayRef<ir::Type>) {
    return parser.builder.create<ir::inst::Nop>(startTok->getRange());
  }

  static ir::Instruction parseLoad(Parser &parser, Token *startTok,
                                   ir::Block *parentBlock,
                                   llvm::ArrayRef<ir::Type>) {
    Parser::RangeHelper rh(parser, startTok);
    auto operand = parser.parseOperand(parentBlock);
    if (parser.diag.hasError())
      return nullptr;

    auto operandType = operand.getType();
    if (!operandType.isa<ir::PointerT>()) {
      parser.report<Parser::ParserDiag::unexpected_type>(
          rh.getRange(), operandType.toString(), "Pointer type");
      return nullptr;
    }

    return parser.builder.create<ir::inst::Load>(rh.getRange(), operand);
  }

  static ir::Instruction parseStore(Parser &parser, Token *startTok,
                                    ir::Block *parentBlock,
                                    llvm::ArrayRef<ir::Type>) {
    Parser::RangeHelper rh(parser, startTok);

    auto value = parser.parseOperand(parentBlock);
    if (parser.diag.hasError())
      return nullptr;

    auto ptr = parser.parseOperand(parentBlock);
    if (parser.diag.hasError())
      return nullptr;

    auto ptrType = ptr.getType();
    if (!ptrType.isa<ir::PointerT>()) {
      parser.report<Parser::ParserDiag::unexpected_type>(
          rh.getRange(), ptrType.toString(), "Pointer type");
      return nullptr;
    }

    if (value.getType().constCanonicalize() !=
        ptrType.cast<ir::PointerT>().getPointeeType().constCanonicalize()) {
      parser.reportError(
          rh.getRange(),
          "Value type does not match pointer's pointee type: " +
              value.getType().toString() + " vs " +
              ptrType.cast<ir::PointerT>().getPointeeType().toString());
      return nullptr;
    }

    return parser.builder.create<ir::inst::Store>(rh.getRange(), value, ptr);
  }

  static ir::Instruction parseCall(Parser &parser, Token *startTok,
                                   ir::Block *parentBlock,
                                   llvm::ArrayRef<ir::Type> types) {
    Parser::RangeHelper rh(parser, startTok);

    auto func = parser.parseOperand(parentBlock);
    if (parser.diag.hasError())
      return nullptr;

    if (!func.getType().isa<ir::PointerT>()) {
      parser.report<Parser::ParserDiag::unexpected_type>(
          rh.getRange().End, func.getType().toString(), "Pointer type");
      return nullptr;
    }

    if (!func.getType()
             .cast<ir::PointerT>()
             .getPointeeType()
             .isa<ir::FunctionT>()) {
      parser.reportError(rh.getRange().End,
                         "Expected a function pointer type, but got: " +
                             func.getType().toString());
      return nullptr;
    }

    auto funcType = func.getType()
                        .cast<ir::PointerT>()
                        .getPointeeType()
                        .cast<ir::FunctionT>();

    if (parser.consume<Token::Tok_lparen>())
      return nullptr;

    llvm::SmallVector<ir::Value> args;
    if (auto peekTok = parser.peekToken(); !peekTok->is<Token::Tok_rparen>()) {
      parser.lexer.rollback(peekTok);

      auto operand = parser.parseOperand(parentBlock);
      if (parser.diag.hasError())
        return nullptr;

      args.emplace_back(operand);

      while (parser.peekToken()->is<Token::Tok_comma>()) {
        parser.nextToken();
        operand = parser.parseOperand(parentBlock);
        if (parser.diag.hasError())
          return nullptr;
        args.emplace_back(operand);
      }
    }

    if (parser.consume<Token::Tok_rparen>())
      return nullptr;

    if (!llvm::equal(
            funcType.getArgTypes(),
            llvm::map_range(args, [](ir::Value op) { return op.getType(); }))) {
      parser.reportError(
          rh.getRange(),
          "Function argument types do not match: expected (" +
              typeRangeToString(funcType.getArgTypes()) + "), got (" +
              typeRangeToString(llvm::map_range(
                  args, [](const ir::Value &v) { return v.getType(); })) +
              ")");
      return nullptr;
    }

    if (!llvm::equal(funcType.getReturnTypes(), types)) {
      parser.reportError(rh.getRange(),
                         "Function return types do not match: expected (" +
                             typeRangeToString(funcType.getReturnTypes()) +
                             "), got (" + typeRangeToString(types) + ")");
      return nullptr;
    }

    return parser.builder.create<ir::inst::Call>(rh.getRange(), func, args);
  }

  static ir::Instruction parseTypeCast(Parser &parser, Token *startTok,
                                       ir::Block *parentBlock,
                                       llvm::ArrayRef<ir::Type>) {
    Parser::RangeHelper rh(parser, startTok);

    auto value = parser.parseOperand(parentBlock);
    if (parser.diag.hasError())
      return nullptr;

    if (parser.consume<Token::Tok_to>())
      return nullptr;

    auto targetType = parser.parseType();
    if (parser.diag.hasError())
      return nullptr;

    return parser.builder.create<ir::inst::TypeCast>(rh.getRange(), value,
                                                     targetType);
  }

  static ir::Instruction parseGep(Parser &parser, Token *startTok,
                                  ir::Block *parentBlock,
                                  llvm::ArrayRef<ir::Type> types) {
    if (types.size() != 1) {
      parser.report<Parser::ParserDiag::unmatched_result_type>(
          startTok->getRange(), "getelementptr", 1, types.size());
      return nullptr;
    }

    Parser::RangeHelper rh(parser, startTok);
    auto base = parser.parseOperand(parentBlock);
    if (parser.diag.hasError())
      return nullptr;

    if (!base.getType().isa<ir::PointerT>()) {
      parser.report<Parser::ParserDiag::unexpected_type>(
          rh.getRange().End, base.getType().toString(), "Pointer type");
      return nullptr;
    }

    if (parser.consume<Token::Tok_offset>())
      return nullptr;

    auto offset = parser.parseOperand(parentBlock);
    if (parser.diag.hasError())
      return nullptr;

    if (!offset.getType().isa<ir::IntT>()) {
      parser.report<Parser::ParserDiag::unexpected_type>(
          rh.getRange().End, offset.getType().toString(), "Integer type");
      return nullptr;
    }

    if (!types[0].isa<ir::PointerT>()) {
      parser.report<Parser::ParserDiag::unexpected_type>(
          rh.getRange(), types[0].toString(), "Pointer type");
      return nullptr;
    }

    auto resultType = types[0].cast<ir::PointerT>();
    return parser.builder.create<ir::inst::Gep>(rh.getRange(), base, offset,
                                                resultType);
  }

#define PARSE_BINARY_OP(name)                                                  \
  static ir::Instruction parse##name(Parser &parser, Token *startTok,          \
                                     ir::Block *parentBlock,                   \
                                     llvm::ArrayRef<ir::Type> types) {         \
    if (types.size() != 1) {                                                   \
      return nullptr;                                                          \
    }                                                                          \
    auto resultType = types[0];                                                \
    return parseBinaryInst(parser, startTok, parentBlock,                      \
                           ir::inst::Binary::OpKind::name, resultType);        \
  }

  PARSE_BINARY_OP(Add);
  PARSE_BINARY_OP(Sub);
  PARSE_BINARY_OP(Mul);
  PARSE_BINARY_OP(Div);
  PARSE_BINARY_OP(Mod);
  PARSE_BINARY_OP(BitAnd);
  PARSE_BINARY_OP(BitOr);
  PARSE_BINARY_OP(BitXor);
  PARSE_BINARY_OP(Shl);
  PARSE_BINARY_OP(Shr);

#undef PARSE_BINARY_OP

  static ir::Instruction parseCmp(Parser &parser, Token *startTok,
                                  ir::Block *parentBlock,
                                  llvm::ArrayRef<ir::Type> types) {
    auto *token = parser.nextToken();
    auto opKind =
        llvm::StringSwitch<ir::inst::Binary::OpKind>(token->getSymbol())
            .Case("eq", ir::inst::Binary::OpKind::Eq)
            .Case("ne", ir::inst::Binary::OpKind::Ne)
            .Case("lt", ir::inst::Binary::OpKind::Lt)
            .Case("le", ir::inst::Binary::OpKind::Le)
            .Case("gt", ir::inst::Binary::OpKind::Gt)
            .Case("ge", ir::inst::Binary::OpKind::Ge)
            .Default(static_cast<ir::inst::Binary::OpKind>(-1));

    if (opKind == static_cast<ir::inst::Binary::OpKind>(-1)) {
      parser.reportError(token->getRange(),
                         "Expected a valid comparison operator (eq, ne, lt, "
                         "le, gt, ge)");
      return nullptr;
    }

    if (types.size() != 1) {
      parser.report<Parser::ParserDiag::unmatched_result_type>(
          startTok->getRange(), "comparison", 1, types.size());
      return nullptr;
    }

    auto resultType = types[0];
    if (!resultType.isa<ir::IntT>() ||
        resultType.cast<ir::IntT>().getBitWidth() != 1) {
      parser.report<Parser::ParserDiag::unexpected_type>(
          startTok->getRange(), resultType.toString(), "Boolean type");
      return nullptr;
    }

    return parseBinaryInst(parser, startTok, parentBlock, opKind, resultType);
  }

  static ir::Instruction parseBinaryInst(Parser &parser, Token *startTok,
                                         ir::Block *parentBlock,
                                         ir::inst::Binary::OpKind opKind,
                                         ir::Type resultType) {
    Parser::RangeHelper rh(parser, startTok);

    auto lhs = parser.parseOperand(parentBlock);
    if (parser.diag.hasError())
      return nullptr;

    auto rhs = parser.parseOperand(parentBlock);
    if (parser.diag.hasError())
      return nullptr;

    if (lhs.getType() != rhs.getType()) {
      parser.reportError(rh.getRange(), "Operand types do not match: " +
                                            lhs.getType().toString() + " vs " +
                                            rhs.getType().toString());
      return nullptr;
    }

    auto inst = parser.builder.create<ir::inst::Binary>(rh.getRange(), lhs, rhs,
                                                        opKind, resultType);
    return inst;
  }

#define PARSE_UNARY_OP(name)                                                   \
  static ir::Instruction parse##name(Parser &parser, Token *startTok,          \
                                     ir::Block *parentBlock,                   \
                                     llvm::ArrayRef<ir::Type> types) {         \
    if (types.size() != 1) {                                                   \
      return nullptr;                                                          \
    }                                                                          \
    return parseUnaryInst(parser, startTok, parentBlock,                       \
                          ir::inst::Unary::OpKind::name, types[0]);            \
  }

  PARSE_UNARY_OP(Plus);
  PARSE_UNARY_OP(Minus);
  PARSE_UNARY_OP(Negate);

#undef PARSE_UNARY_OP

  static ir::Instruction parseUnaryInst(Parser &parser, Token *startTok,
                                        ir::Block *parentBlock,
                                        ir::inst::Unary::OpKind opKind,
                                        ir::Type resultType) {
    Parser::RangeHelper rh(parser, startTok);

    auto value = parser.parseOperand(parentBlock);
    if (parser.diag.hasError())
      return nullptr;

    auto inst =
        parser.builder.create<ir::inst::Unary>(rh.getRange(), value, opKind);
    if (inst.getType().constCanonicalize() != resultType.constCanonicalize()) {
      parser.report<Parser::ParserDiag::unmatched_result_type>(
          inst.getRange(), inst.getType().toString(), resultType.toString());
      return nullptr;
    }
    return inst;
  }

  static ir::BlockExit parseJump(Parser &parser, Token *startTok,
                                 ir::Block *parentBlock) {
    Parser::RangeHelper rh(parser, startTok);

    auto jumpArg = parser.parseJumpArg(parentBlock);
    if (parser.diag.hasError())
      return nullptr;

    auto jump = parser.builder.create<ir::inst::Jump>(rh.getRange(), jumpArg);
    return jump;
  }

  static ir::BlockExit parseBranch(Parser &parser, Token *startTok,
                                   ir::Block *parentBlock) {
    Parser::RangeHelper rh(parser, startTok);

    auto condition = parser.parseOperand(parentBlock);
    if (parser.diag.hasError())
      return nullptr;

    if (!condition.getType().isa<ir::IntT>() ||
        condition.getType().cast<ir::IntT>().getBitWidth() != 1) {
      parser.report<Parser::ParserDiag::unexpected_type>(
          rh.getRange().End, condition.getType().toString(), "Boolean type");
      return nullptr;
    }

    if (parser.consume<Token::Tok_comma>())
      return nullptr;

    auto trueJumpArg = parser.parseJumpArg(parentBlock);
    if (parser.diag.hasError())
      return nullptr;

    if (parser.consume<Token::Tok_comma>())
      return nullptr;

    auto falseJumpArg = parser.parseJumpArg(parentBlock);
    if (parser.diag.hasError())
      return nullptr;

    auto branch = parser.builder.create<ir::inst::Branch>(
        rh.getRange(), condition, trueJumpArg, falseJumpArg);
    return branch;
  }

  static ir::BlockExit parseSwitch(Parser &parser, Token *startTok,
                                   ir::Block *parentBlock) {
    Parser::RangeHelper rh(parser, startTok);

    auto value = parser.parseOperand(parentBlock);
    if (parser.diag.hasError())
      return nullptr;

    if (parser.consume<Token::Tok_default>())
      return nullptr;

    auto defaultJumpArg = parser.parseJumpArg(parentBlock);
    if (parser.diag.hasError())
      return nullptr;

    llvm::SmallVector<ir::Value> caseValues;
    llvm::SmallVector<ir::JumpArgState> cases;

    if (parser.consume<Token::Tok_lbracket>())
      return nullptr;

    while (true) {
      if (parser.consumeIf<Token::Tok_rbracket>())
        break;

      auto caseValue = parser.parseOperand(parentBlock);
      if (parser.diag.hasError())
        return nullptr;

      if (caseValue.getType() != value.getType()) {
        parser.report<Parser::ParserDiag::unexpected_type>(
            rh.getRange().End, caseValue.getType().toString(),
            value.getType().toString());
        return nullptr;
      }

      auto caseJumpArg = parser.parseJumpArg(parentBlock);
      if (parser.diag.hasError())
        return nullptr;

      caseValues.emplace_back(caseValue);
      cases.emplace_back(caseJumpArg);
    };

    auto switchInst = parser.builder.create<ir::inst::Switch>(
        rh.getRange(), value, caseValues, cases, defaultJumpArg);

    return switchInst;
  }

  static ir::BlockExit parseReturn(Parser &parser, Token *startTok,
                                   ir::Block *parentBlock) {
    Parser::RangeHelper rh(parser, startTok);

    auto peekTok = parser.peekConstantToken();
    if (peekTok->is<Token::Tok_unknown>()) {
      parser.lexer.rollback(peekTok);
      peekTok = parser.peekRidToken();
      if (peekTok->is<Token::Tok_unknown, Token::Tok_identifier>()) {
        parser.lexer.rollback(peekTok);
        return parser.builder.create<ir::inst::Return>(rh.getRange());
      }
    }

    parser.lexer.rollback(peekTok);

    ir::Value value = parser.parseOperand(parentBlock);
    if (parser.diag.hasError())
      return nullptr;

    llvm::SmallVector<ir::Value> values(1, value);
    while (parser.consumeIf<Token::Tok_comma>()) {
      auto nextValue = parser.parseOperand(parentBlock);
      if (parser.diag.hasError())
        return nullptr;

      values.emplace_back(nextValue);
    }

    auto returnInst =
        parser.builder.create<ir::inst::Return>(rh.getRange(), values);
    return returnInst;
  }

  static ir::BlockExit parseUnreachable(Parser &parser, Token *startTok,
                                        ir::Block *parentBlock) {
    Parser::RangeHelper rh(parser, startTok);

    auto unreachableInst =
        parser.builder.create<ir::inst::Unreachable>(rh.getRange());
    return unreachableInst;
  }
};
} // namespace detail

ir::Instruction Parser::parseInstructionInner(ir::Block *parentBlock,
                                              llvm::ArrayRef<ir::Type> types) {
  auto *token = nextToken();
  if (expect<Token::Tok_identifier>(token))
    return nullptr;
  auto instName = token->getSymbol();

  auto func =
      llvm::StringSwitch<ir::Instruction (*)(
          Parser &, Token *, ir::Block *, llvm::ArrayRef<ir::Type>)>(instName)

  // clang-format off
#define CASE(symbol, op) .Case(symbol, &detail::ParserDetail::parse##op)

                  CASE("nop", Nop)
                  CASE("load", Load)
                  CASE("store", Store)
                  CASE("call", Call)
                  CASE("typecast", TypeCast)
                  CASE("add", Add)
                  CASE("sub", Sub)
                  CASE("mul", Mul)
                  CASE("div", Div)
                  CASE("mod", Mod)
                  CASE("and", BitAnd)
                  CASE("or", BitOr)
                  CASE("xor", BitXor)
                  CASE("shl", Shl) 
                  CASE("shr", Shr)
                  CASE("cmp", Cmp)
                  CASE("plus", Plus) 
                  CASE("minus", Minus)
                  CASE("negate", Negate)
                  CASE("getelementptr", Gep)
#undef CASE

                  .Default(nullptr);
  // clang-format on

  if (!func) {
    reportError(token->getRange(),
                ("Unknown instruction: '" + instName + "'").str());
    return nullptr;
  }

  return func(*this, token, parentBlock, types);
}

ir::BlockExit Parser::parseBlockExit(Token *firstToken,
                                     ir::Block *parentBlock) {
  // jump, branch, switch, ret, unreachable

  auto parseFunc =
      llvm::StringSwitch<ir::BlockExit (*)(Parser &, Token *, ir::Block *)>(
          firstToken->getSymbol())
#define CASE(symbol, op) .Case(symbol, &detail::ParserDetail::parse##op)
      // clang-format off
                  CASE("j", Jump)
                  CASE("br", Branch)
                  CASE("switch", Switch)
                  CASE("ret", Return)
                  CASE("unreachable", Unreachable)
     
  #undef CASE 
                  .Default(nullptr);
  // clang-format on

  if (!parseFunc)
    return nullptr;

  auto blockExit = parseFunc(*this, firstToken, parentBlock);
  if (diag.hasError())
    return nullptr;

  return blockExit;
}

ir::Type Parser::parseType() { return parsePointerType(); }

ir::Type Parser::parsePointerType() {

  auto constType = parseConstType();
  if (diag.hasError())
    return nullptr;

  while (consumeIf<Token::Tok_asterisk>()) {
    auto peekTok = peekToken();
    bool isConst = false;
    if (peekTok->is<Token::Tok_const>()) {
      nextToken(); // consume const
      isConst = true;
    } else {
      lexer.rollback(peekTok);
    }
    constType = ir::PointerT::get(context, constType, isConst);
  }

  // Next token of type can be non-general, so we need to rollback
  auto peekToken = lexer.peekToken();
  lexer.rollback(peekToken);

  return constType;
}

ir::Type Parser::parseConstType() {
  // 'const' raw_type
  // i1, i8, i16, i32, i64
  // u1, u8, u16, u32, u64
  // f32, f64,
  // struct identifier
  // '[' 'ret' ':' type (',' type)* ',' 'params' '(' type (',' type)* ')' ']'
  // '[' number 'x' type ']'

  bool isConst = false;
  if (consumeIf<Token::Tok_const>()) {
    isConst = true;
  }

  auto rawType = parseRawType();
  if (diag.hasError())
    return nullptr;

  if (isConst)
    rawType = ir::ConstQualifier::get(context, rawType);

  return rawType;
}

ir::Type Parser::parseRawType() {
  auto *token = nextToken();
  ir::Type result;
  switch (token->getKind()) {

    // clang-format off
#define SIGNED_INT_CASE(bits)                                                  \
  case Token::Tok_i##bits:                                                     \
    result = ir::IntT::get(context, bits, true);                               \
    break;
      SIGNED_INT_CASE(1)
      SIGNED_INT_CASE(8)
      SIGNED_INT_CASE(16)
      SIGNED_INT_CASE(32)
      SIGNED_INT_CASE(64)

#undef SIGNED_INT_CASE

#define UNSIGNED_INT_CASE(bits)                                                \
  case Token::Tok_u##bits:                                                     \
    result = ir::IntT::get(context, bits, false);                              \
    break;

      UNSIGNED_INT_CASE(1)
      UNSIGNED_INT_CASE(8)
      UNSIGNED_INT_CASE(16)
      UNSIGNED_INT_CASE(32)
      UNSIGNED_INT_CASE(64)

#undef UNSIGNED_INT_CASE

    // clang-format on
  case Token::Tok_f32:
    result = ir::FloatT::get(context, 32);
    break;
  case Token::Tok_f64:
    result = ir::FloatT::get(context, 64);
    break;

  case Token::Tok_unit:
    result = ir::UnitT::get(context);
    break;

  case Token::Tok_struct: {
    auto *nameTok = nextToken();
    if (expect<Token::Tok_identifier>(nameTok))
      return nullptr;

    auto name = nameTok->getSymbol();
    result = ir::NameStruct::get(context, name);
    break;
  }

  case Token::Tok_lbracket: {
    if (!peekToken()->is<Token::Tok_ret>()) {
      // '[' number 'x' type ']'
      auto numberToken = nextToken();
      if (expect<Token::Tok_integer>(numberToken))
        return nullptr;

      std::size_t arraySize;
      auto failed = numberToken->getSymbol().getAsInteger(10, arraySize);
      assert(!failed && "Expected a valid integer for array size. It is "
                        "guaranteed by lexer.");
      (void)failed;

      auto token = lexer.tryLex("x", Token::Tok_x);
      if (!token->is<Token::Tok_x>()) {
        reportError(token->getRange(), "Expected 'x' token for array type");
        return nullptr;
      }

      auto type = parseType();
      if (diag.hasError())
        return nullptr;
      if (consume<Token::Tok_rbracket>())
        return nullptr;

      result = ir::ArrayT::get(context, arraySize, type);
      break;
    }

    // '[' 'ret' ':' type (',' type)* ',' 'params' '(' type (',' type)* ')' ']'
    if (consumeStream<Token::Tok_ret, Token::Tok_colon>())
      return nullptr;

    auto type = parseType();

    llvm::SmallVector<ir::Type> retTypes{type};
    while (consumeIf<Token::Tok_comma>()) {
      type = parseType();
      if (diag.hasError())
        return nullptr;
      retTypes.emplace_back(type);
    }

    if (consumeStream<Token::Tok_params, Token::Tok_colon, Token::Tok_lparen>())
      return nullptr;

    llvm::SmallVector<ir::Type> paramTypes;
    if (!consumeIf<Token::Tok_rparen>()) {
      auto type = parseType();
      if (diag.hasError())
        return nullptr;
      paramTypes.emplace_back(type);
      while (consumeIf<Token::Tok_comma>()) {
        type = parseType();
        if (diag.hasError())
          return nullptr;
        paramTypes.emplace_back(type);
      }
      if (consume<Token::Tok_rparen>())
        return nullptr;
    }

    if (consume<Token::Tok_rbracket>())
      return nullptr;

    result = ir::FunctionT::get(context, retTypes, paramTypes);
    break;
  }

  default: {
    reportError(token->getRange(), "Unexpected token for parsing type: " +
                                       token->getSymbol().str());
    return nullptr;
  }
  }

  // Next token of type can be non-general, so we need to rollback
  auto peekToken = lexer.peekToken();
  lexer.rollback(peekToken);

  return result;
}

ir::Value Parser::parseOperand(ir::Block *parentBlock) {
  auto *constToken = nextConstantToken();
  if (!constToken->is<Token::Tok_unknown>()) {
    RangeHelper rh(*this, constToken);
    auto constantAttr = parseConstant(parentBlock, constToken);
    if (diag.hasError())
      return nullptr;

    ir::IRBuilder::InsertionGuard guard(
        builder, parentBlock->getParentIR()->getConstantBlock());

    return builder.create<ir::inst::Constant>(rh.getRange(), constantAttr)
        .getResult();
  } else {
    // If the token is not a constant, it should be a register id
    lexer.rollback(constToken);
  }
  auto *ridToken = peekRidToken();
  RangeHelper rh(*this, ridToken);
  if (ridToken->is<Token::Tok_unknown>()) {
    reportError(rh.getRange(), "Expected a valid register id or constant");
    return nullptr;
  }

  auto resultOpt = parseRegisterId(parentBlock);
  if (diag.hasError())
    return nullptr;

  auto [regId, type, name] = *resultOpt;
  if (!name.empty()) {
    reportError(rh.getRange(), "Parsing operand with name is not supported");
    return nullptr;
  }

  auto it = registerMap.find(regId);

  ir::Value result;
  if (it == registerMap.end()) {
    // create temp operation by using unresolved instruction in hidden block

    ir::IRBuilder::InsertionGuard guard(
        builder, parentBlock->getParentFunction()->getUnresolvedBlock());

    auto unresolvedInst =
        builder.create<ir::inst::Unresolved>(rh.getRange(), type);
    registerMap.try_emplace(regId, ir::Value(unresolvedInst));
    result = unresolvedInst;
  } else {
    // use existing operation
    result = it->second;
    if (result.getType().constCanonicalize() != type.constCanonicalize()) {
      report<ParserDiag::unmatched_type_with_declaration>(
          rh.getRange(), result.getType().toString(), type.toString());
      report<ParserDiag::declaration_info>(result.getInstruction()->getRange());
      return {};
    }
  }

  return result;
}

ir::JumpArgState Parser::parseJumpArg(ir::Block *parentBlock) {
  // b number '(' operand (',' operand)* ')'

  auto *token = nextRidToken();
  RangeHelper rh(*this, token);

  if (expect<Token::Tok_block_id>(token))
    return {};

  if (consume<Token::Tok_lparen>())
    return {};

  auto peekTok = peekToken();
  llvm::SmallVector<ir::Value> args;
  if (!peekTok->is<Token::Tok_rparen>()) {
    lexer.rollback(peekTok);

    auto operand = parseOperand(parentBlock);
    if (diag.hasError())
      return {};

    args.emplace_back(operand);

    while (peekToken()->is<Token::Tok_comma>()) {
      nextToken();
      operand = parseOperand(parentBlock);
      if (diag.hasError())
        return {};

      args.emplace_back(operand);
    }

    if (consume<Token::Tok_rparen>())
      return {};
  } else {
    nextToken();
  }

  auto blockId = token->getNumberFromId();
  auto block = parentBlock->getParentFunction()->getBlockById(blockId);
  if (!block)
    block = parentBlock->getParentFunction()->addBlock(blockId);

  return {block, args};
}

ir::ConstantAttr Parser::parseConstant(ir::Block *parentBlock,
                                       Token *startToken) {
  bool isMinus = false;
  Token *token = startToken;
  if (startToken->is<Token::Tok_minus, Token::Tok_plus>()) {
    isMinus = startToken->is<Token::Tok_minus>();
    token = nextConstantToken();
    if (token->is<Token::Tok_unknown>()) {
      reportError(startToken->getRange(),
                  "Invalid token for constant parsing: " +
                      startToken->getSymbol().str());
      return nullptr;
    }
    auto constant = parseConstant(parentBlock, token);
    if (diag.hasError())
      return nullptr;
    return isMinus ? constant.insertMinus() : constant;
  }

  RangeHelper rh(*this, token);

  if (token->is<Token::Tok_integer>()) {
    std::size_t numberValue;
    auto value = token->getSymbol().getAsInteger(10, numberValue);
    if (value) {
      reportError(token->getRange(),
                  "Invalid integer constant: " + token->getSymbol().str());
      return nullptr;
    }

    if (consume<Token::Tok_colon>())
      return nullptr;

    auto type = parseType();
    if (diag.hasError())
      return nullptr;

    if (!type.isa<ir::IntT>()) {
      report<ParserDiag::unexpected_type>(rh.getRange(), type.toString(),
                                          "Integer type");
      return nullptr;
    }

    return ir::ConstantIntAttr::get(context, numberValue,
                                    type.cast<ir::IntT>().getBitWidth(),
                                    type.cast<ir::IntT>().isSigned());
  } else if (token->is<Token::Tok_float>()) {
    if (consume<Token::Tok_colon>())
      return nullptr;
    auto type = parseType();
    if (diag.hasError())
      return nullptr;

    if (!type.isa<ir::FloatT>()) {
      report<ParserDiag::unexpected_type>(rh.getRange(), type.toString(),
                                          "Float type");
      return nullptr;
    }

    if (type.cast<ir::FloatT>().getBitWidth() == 32) {
      llvm::APFloat floatValue(llvm::APFloat::IEEEsingle());
      auto err = floatValue.convertFromString(
          token->getSymbol(), llvm::APFloat::rmNearestTiesToEven);
      if (!err) {
        reportError(token->getRange(),
                    "Invalid float constant: " + token->getSymbol().str());
        return nullptr;
      }

    } else {
      llvm::APFloat floatValue(llvm::APFloat::IEEEdouble());
      auto err = floatValue.convertFromString(
          token->getSymbol(), llvm::APFloat::rmNearestTiesToEven);

      if (!err) {
        reportError(token->getRange(),
                    "Invalid double constant: " + token->getSymbol().str());
        return nullptr;
      }
    }

    // Create a string float constant instead of float constant which is using
    // `llvm::APFloat`. This is because `llvm::APFloat` can't print original
    // text
    return ir::ConstantStringFloatAttr::get(context, token->getSymbol(),
                                            type.cast<ir::FloatT>());
  } else if (token->is<Token::Tok_undef>()) {
    if (consume<Token::Tok_colon>())
      return nullptr;
    auto type = parseType();
    if (diag.hasError())
      return nullptr;

    return ir::ConstantUndefAttr::get(context, type);
  } else if (token->is<Token::Tok_unit>()) {
    if (consume<Token::Tok_colon>())
      return nullptr;

    auto type = parseType();
    if (diag.hasError())
      return nullptr;

    if (!type.isa<ir::UnitT>()) {
      report<ParserDiag::unexpected_type>(rh.getRange(), type.toString(),
                                          "Unit type");
      return nullptr;
    }

    return ir::ConstantUnitAttr::get(context);
  } else if (token->is<Token::Tok_global_variable>()) {
    if (consume<Token::Tok_colon>())
      return nullptr;

    auto type = parseType();
    if (diag.hasError())
      return nullptr;

    if (!type.isa<ir::PointerT>()) {
      report<ParserDiag::unexpected_type>(rh.getRange(), type.toString(),
                                          "Pointer type");
      return nullptr;
    }

    auto globalVarName = token->getSymbol().drop_front(); // remove '@'
    return ir::ConstantVariableAttr::get(context, globalVarName, type);
  }

  reportError(token->getRange(), "Invalid token for constant parsing: " +
                                     token->getSymbol().str());
  return nullptr;
}

ir::InitializerAttr Parser::parseInitializerAttr(ir::IR *program) {
  // '{' initializer (',' initializer)* '}'
  // ('-' | '+') initializer
  // '(' initializer ')'
  // integer
  // float

  auto *token = nextInitializerToken();
  RangeHelper rh(*this, token);
  switch (token->getKind()) {
  case Token::Tok_lbrace: {
    auto initializer = parseInitializerAttr(program);
    if (diag.hasError())
      return nullptr;
    llvm::SmallVector<ir::InitializerAttr> initializers{initializer};

    while (peekInitializerToken()->is<Token::Tok_comma>()) {
      nextInitializerToken();
      initializer = parseInitializerAttr(program);
      if (diag.hasError())
        return nullptr;

      initializers.emplace_back(initializer);
    }
    if (consume<Token::Tok_rbrace>())
      return nullptr;
    return ir::ASTInitializerList::get(context, rh.getRange(), initializers);
  }
  case Token::Tok_minus: {
    auto initializer = parseInitializerAttr(program);
    if (diag.hasError())
      return nullptr;
    return ir::ASTUnaryOp::get(context, rh.getRange(),
                               ir::ASTUnaryOp::OpKind::Minus, initializer);
  }
  case Token::Tok_plus: {
    auto initializer = parseInitializerAttr(program);
    if (diag.hasError())
      return nullptr;
    return ir::ASTUnaryOp::get(context, rh.getRange(),
                               ir::ASTUnaryOp::OpKind::Plus, initializer);
  }
  case Token::Tok_lparen: {
    // '(' initializer ')'
    auto initializer = parseInitializerAttr(program);
    if (diag.hasError())
      return nullptr;

    if (consume<Token::Tok_rparen>())
      return nullptr;

    return ir::ASTGroupOp::get(context, rh.getRange(), initializer);
  }
  case Token::Tok_ast_integer: {
    auto symbol = token->getSymbol();
    auto integerBase = ir::ASTInteger::IntegerBase::Decimal;

    if (symbol.size() > 2) {
      if (symbol[0] == '0')
        integerBase = ir::ASTInteger::IntegerBase::Octal;
      auto prefix = symbol.substr(0, 2);
      if (prefix == "0x" || prefix == "0X") {
        integerBase = ir::ASTInteger::IntegerBase::Hexadecimal;
        symbol = symbol.substr(2);
      }
    }

    auto suffix = ir::ASTInteger::Suffix::Int;
    if (symbol.back() == 'l') {
      suffix = ir::ASTInteger::Suffix::Long_l;
      symbol = symbol.drop_back();
    } else if (symbol.back() == 'L') {
      suffix = ir::ASTInteger::Suffix::Long_L;
      symbol = symbol.drop_back();
    }

    return ir::ASTInteger::get(context, rh.getRange(), integerBase, symbol,
                               suffix);
  }
  case Token::Tok_ast_float: {
    auto symbol = token->getSymbol();
    auto suffix = ir::ASTFloat::Suffix::Double;
    if (symbol.back() == 'f') {
      suffix = ir::ASTFloat::Suffix::Float_f;
      symbol = symbol.drop_back();
    } else if (symbol.back() == 'F') {
      suffix = ir::ASTFloat::Suffix::Float_F;
      symbol = symbol.drop_back();
    }
    return ir::ASTFloat::get(context, rh.getRange(), symbol, suffix);
  }
  default:
    reportError(token->getRange(), "Invalid token for initializer parsing: " +
                                       token->getSymbol().str());
    return nullptr;
  }
}

static llvm::SourceMgr::DiagKind parserDiagKinds[] = {
#define DIAG(Name, Message, Kind) llvm::SourceMgr::DK_##Kind,
#include "kecc/parser/ParserDiag.def"
};

static const char *parserDiagMessages[] = {
#define DIAG(Name, Message, Kind) Message,
#include "kecc/parser/ParserDiag.def"
};

llvm::SourceMgr::DiagKind Parser::ParserDiag::getDiagKind(Diag diag) {
  return parserDiagKinds[static_cast<size_t>(diag)];
}

const char *Parser::ParserDiag::getDiagMessage(Diag diag) {
  return parserDiagMessages[static_cast<size_t>(diag)];
}

} // namespace kecc
