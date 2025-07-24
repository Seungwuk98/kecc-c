#ifndef KECC_IR_PATTERN_MATCH_H
#define KECC_IR_PATTERN_MATCH_H

#include "kecc/ir/IRBuilder.h"
#include "kecc/ir/Instruction.h"
#include "kecc/ir/Module.h"
#include "kecc/ir/WalkSupport.h"
#include "kecc/utils/LogicalResult.h"
#include "mlir/Support/TypeID.h"
#include "llvm/ADT/DenseSet.h"
#include <cstdint>

namespace kecc::ir {

class TransactionInstState {
public:
  InstructionStorage *getInst() const { return inst; }

  static TransactionInstState create(InstructionStorage *inst);
  void rollback();

private:
  InstructionStorage *inst;
  llvm::SMRange range;
  llvm::SmallVector<Value> operands;
  llvm::SmallVector<std::pair<Type, std::string>> types;
  llvm::SmallVector<Attribute> attributes;
  llvm::SmallVector<JumpArgState> jumpArgs;
  Block *parentBlock;
  AbstractInstruction *abstractInst;
};

class TransactionFunctionState {
public:
  Function *getFunction() const { return function; }

  static TransactionFunctionState create(Function *function);
  void rollback();

private:
  Function *function;
  std::string name;
  Type functionType;
  int entryBid;
};

class BlockAction {
public:
  enum Kind {
    Create,
    Remove,
  };

  BlockAction(Kind kind, Block *block) : kind(kind), block(block) {}

  static BlockAction created(Block *block) {
    return BlockAction(Kind::Create, block);
  }

  static BlockAction removed(Block *block) {
    return BlockAction(Kind::Remove, block);
  }

  Kind getKind() const { return kind; }
  Block *getBlock() const { return block; }

private:
  Kind kind;
  Block *block;
};

class FunctionAction {
public:
  enum Kind {
    Create,
    Remove,
    Modified,
  };

  FunctionAction(
      Kind kind, Function *function,
      std::unique_ptr<TransactionFunctionState> transactionState = nullptr)
      : kind(kind), function(function),
        transactionState(std::move(transactionState)) {}

  static FunctionAction created(Function *function) {
    return FunctionAction(Kind::Create, function);
  }

  static FunctionAction removed(Function *function) {
    return FunctionAction(Kind::Remove, function);
  }

  static FunctionAction modifyStart(Function *function) {
    return FunctionAction(Kind::Modified, function,
                          std::make_unique<TransactionFunctionState>(
                              TransactionFunctionState::create(function)));
  }

  Kind getKind() const { return kind; }
  Function *getFunction() const { return function; }
  TransactionFunctionState *getTransactionState() const {
    return transactionState.get();
  }

private:
  Kind kind;
  Function *function;
  std::unique_ptr<TransactionFunctionState> transactionState;
};

struct IRRewriterImpl;
class IRRewriter : public IRBuilder {
public:
  IRRewriter(Module *module, IRContext *context);
  ~IRRewriter();

  struct State {
    size_t transactionCount = 0;
    size_t createdInstsCount = 0;
    size_t removedInstsCount = 0;
    size_t functionActionCount = 0;
    size_t blockActionCount = 0;
    size_t replaceMapCount = 0;
  };

  template <typename Inst, typename... Args>
  Inst create(llvm::SMRange range, Args &&...args) {
    Inst inst = IRBuilder::create<Inst>(range, std::forward<Args>(args)...);
    notifyInstCreated(inst.getStorage());
    return inst;
  }

  void notifyInstCreated(InstructionStorage *inst);
  void removeInst(InstructionStorage *inst);
  void notifyStartUpdate(InstructionStorage *inst);

  void notifyFunctionCreated(Function *func);
  void notifyFunctionRemoved(Function *func);
  void notifyStartUpdate(Function *func);

  void notifyBlockCreated(Block *block);
  void notifyBlockRemoved(Block *block);

  State getCurrentState() const;
  void resetToState(const State &state);

  void applyRewrite();
  void discardRewrite();

  void replaceInst(InstructionStorage *inst, llvm::ArrayRef<Value> values);
  void replaceValue(Value from, Value to);

  IRRewriterImpl *getImpl() const { return impl.get(); }

  bool isInstIgnored(InstructionStorage *inst) const;

  std::pair<llvm::SmallVector<Value>, llvm::SmallVector<JumpArgState>>
  remapping(llvm::ArrayRef<Operand> values, llvm::ArrayRef<JumpArg *> jumpArgs);

private:
  Module *module;
  std::unique_ptr<IRRewriterImpl> impl;
};

class PatternId {
public:
  enum class Kind {
    General,
    Instruction,
    Trait,
    Interface,
  };

  PatternId(Kind kind, TypeID id) : kind(kind), id(id) {}
  // for instruction
  template <typename Inst>
    requires std::is_base_of_v<Instruction, Inst>
  static PatternId get() {
    return PatternId(Kind::Instruction, TypeID::get<Inst>());
  }

  // for interface
  template <typename InstInterface> static PatternId get() {
    return PatternId(Kind::Interface, TypeID::get<InstInterface>());
  }

  template <template <typename> typename Trait> static PatternId get() {
    return PatternId(Kind::Trait, TypeID::get<Trait>());
  }

  static PatternId getGeneral() {
    return PatternId(Kind::General, TypeID::getFromOpaquePointer(nullptr));
  }

  bool operator==(const PatternId &other) const {
    return kind == other.kind && id == other.id;
  }
  bool operator!=(const PatternId &other) const { return !(*this == other); }

  Kind getKind() const { return kind; }
  TypeID getId() const { return id; }

  bool isGeneral() const { return kind == Kind::General; }
  bool isInstruction() const { return kind == Kind::Instruction; }
  bool isTrait() const { return kind == Kind::Trait; }
  bool isInterface() const { return kind == Kind::Interface; }

  friend llvm::hash_code hash_value(const PatternId &id) {
    return llvm::hash_combine(static_cast<unsigned>(id.kind), id.id);
  }

private:
  Kind kind;
  TypeID id;
};
} // namespace kecc::ir

namespace llvm {

template <> struct DenseMapInfo<kecc::ir::PatternId> {
  using PatternId = kecc::ir::PatternId;

  static PatternId getEmptyKey() {
    return PatternId(PatternId::Kind::Instruction,
                     mlir::TypeID::getFromOpaquePointer(
                         llvm::DenseMapInfo<void *>::getEmptyKey()));
  }

  static PatternId getTombstoneKey() {
    return PatternId(PatternId::Kind::Instruction,
                     mlir::TypeID::getFromOpaquePointer(
                         llvm::DenseMapInfo<void *>::getTombstoneKey()));
  }

  static unsigned getHashValue(const PatternId &id) { return hash_value(id); }

  static bool isEqual(const PatternId &lhs, const PatternId &rhs) {
    return lhs == rhs;
  }
};

} // namespace llvm

namespace kecc::ir {

class Pattern {
public:
  Pattern(std::int64_t benefit, PatternId id)
      : benefit(benefit), patternId(id) {}

  virtual ~Pattern() = default;
  virtual utils::LogicalResult match(InstructionStorage *inst) {
    llvm_unreachable("Pattern::match should be overridden");
  }
  virtual utils::LogicalResult rewrite(IRRewriter &rewriter,
                                       InstructionStorage *inst) {
    llvm_unreachable("Pattern::rewrite should be overridden");
  }
  virtual utils::LogicalResult matchAndRewrite(IRRewriter &rewriter,
                                               InstructionStorage *inst) {
    auto result = match(inst);
    if (result.succeeded())
      return rewrite(rewriter, inst);
    return result;
  }

  std::int64_t getBenefit() const { return benefit; }
  PatternId getPatternId() const { return patternId; }

private:
  std::int64_t benefit;
  PatternId patternId;
};

template <typename ConcreteInst> class InstPattern : public Pattern {
public:
  InstPattern(std::int64_t benefit = 1)
      : Pattern(benefit, PatternId::get<ConcreteInst>()) {}

  utils::LogicalResult match(InstructionStorage *inst) override final {
    auto conInst = inst->getDefiningInst<ConcreteInst>();
    return match(conInst);
  }
  utils::LogicalResult rewrite(IRRewriter &rewriter,
                               InstructionStorage *inst) override final {
    auto conInst = inst->getDefiningInst<ConcreteInst>();
    return rewrite(rewriter, conInst);
  }
  utils::LogicalResult
  matchAndRewrite(IRRewriter &rewriter,
                  InstructionStorage *inst) override final {
    auto conInst = inst->getDefiningInst<ConcreteInst>();
    if (!conInst)
      return utils::LogicalResult::failure(); // fail
    return matchAndRewrite(rewriter, conInst);
  }
  virtual utils::LogicalResult match(ConcreteInst inst) {
    llvm_unreachable("Pattern::match should be overridden");
  };
  virtual utils::LogicalResult rewrite(IRRewriter &rewriter,
                                       ConcreteInst inst) {
    llvm_unreachable("Pattern::rewrite should be overridden");
  };
  virtual utils::LogicalResult matchAndRewrite(IRRewriter &rewriter,
                                               ConcreteInst inst) {
    auto result = match(inst);
    if (result.succeeded())
      return rewrite(rewriter, inst);
    return result;
  }
};

template <typename ConcreteInterface> class InterfacePattern : public Pattern {
public:
  InterfacePattern(std::int64_t benefit)
      : Pattern(benefit, PatternId::get<ConcreteInterface>()) {}

  utils::LogicalResult match(InstructionStorage *inst) override final {
    auto conInterface = llvm::cast<ConcreteInterface>(inst);
    return match(conInterface);
  }

  utils::LogicalResult rewrite(IRRewriter &rewriter,
                               InstructionStorage *inst) override final {
    auto conInterface = llvm::cast<ConcreteInterface>(inst);
    return rewrite(conInterface, rewriter);
  }
  utils::LogicalResult
  matchAndRewrite(IRRewriter &rewriter,
                  InstructionStorage *inst) override final {
    auto conInterface = llvm::dyn_cast<ConcreteInterface>(inst);
    if (!conInterface)
      return utils::LogicalResult::failure(); // fail
    return matchAndRewrite(conInterface, rewriter);
  }

  virtual utils::LogicalResult match(ConcreteInterface inst) {
    llvm_unreachable("Pattern::match should be overridden");
  }
  virtual utils::LogicalResult rewrite(ConcreteInterface inst,
                                       IRRewriter &rewriter) {
    llvm_unreachable("Pattern::rewrite should be overridden");
  }
  virtual utils::LogicalResult matchAndRewrite(ConcreteInterface inst,
                                               IRRewriter &rewriter) {
    auto result = match(inst);
    if (result.succeeded())
      return rewrite(inst, rewriter);
    return result;
  }
};

template <template <typename> typename Trait>
class TraitPattern : public Pattern {
public:
  TraitPattern(std::int64_t benefit)
      : Pattern(benefit, PatternId::get<Trait>()) {}
};

template <typename ConcreteInst>
class InstConversionPattern : public InstPattern<ConcreteInst> {
public:
  using Adaptor = typename ConcreteInst::Adaptor;

  InstConversionPattern(std::int64_t benefit = 1)
      : InstPattern<ConcreteInst>(benefit) {}

  utils::LogicalResult match(ConcreteInst inst) override final {
    llvm_unreachable("Implement MatchAndRewrite in derived class");
  }

  utils::LogicalResult rewrite(IRRewriter &rewriter,
                               ConcreteInst inst) override final {
    llvm_unreachable("Implement MatchAndRewrite in derived class");
  }

  utils::LogicalResult matchAndRewrite(IRRewriter &rewriter,
                                       ConcreteInst inst) override final {
    auto [operands, jumpArgs] =
        rewriter.remapping(inst->getOperands(), inst->getJumpArgs());
    Adaptor adaptor(operands, jumpArgs);
    return this->matchAndRewrite(rewriter, adaptor, inst);
  }

  virtual utils::LogicalResult
  matchAndRewrite(IRRewriter &rewriter, Adaptor adaptor, ConcreteInst inst) = 0;
};

class PatternSet {
public:
  template <typename... Patterns, typename... Args>
  void addPatterns(Args &&...args) {
    (addPattern<Patterns>(std::forward<Args>(args)...), ...);
  }

  llvm::ArrayRef<std::unique_ptr<Pattern>> getPatterns() const {
    return patterns;
  }

private:
  template <typename ConcretePattern, typename... Args>
  void addPattern(Args &&...args) {
    patterns.emplace_back(
        std::make_unique<ConcretePattern>(std::forward<Args>(args)...));
  }

  std::vector<std::unique_ptr<Pattern>> patterns;
};

class OrderedPatternSet {
public:
  OrderedPatternSet(const PatternSet &set);

  template <typename Inst> void getInstPattern() const {
    return getInstPattern(TypeID::get<Inst>());
  };
  llvm::ArrayRef<Pattern *> getInstPattern(TypeID typeId) const;

  template <template <typename> typename Trait>
  llvm::ArrayRef<Pattern *> getTraitPattern() const {
    return getTraitPattern(TypeID::get<Trait>());
  }
  llvm::ArrayRef<Pattern *> getTraitPattern(TypeID typeId) const;

  template <typename InstInterface>
  llvm::ArrayRef<Pattern *> getInterfacePattern() const {
    return getInterfacePattern(TypeID::get<InstInterface>());
  }

  llvm::ArrayRef<Pattern *> getInterfacePattern(TypeID typeId) const;

  llvm::SmallVector<Pattern *>
  getAppliablePatterns(InstructionStorage *inst) const;

  void pushGeneralPattern(Pattern *pattern);
  void pushInstPattern(TypeID id, Pattern *pattern);
  void pushTraitPattern(TypeID id, Pattern *pattern);
  void pushInterfacePattern(TypeID id, Pattern *pattern);

private:
  llvm::SmallVector<Pattern *> generalPatterns;
  llvm::DenseMap<TypeID, llvm::SmallVector<Pattern *>> instPatternMap;
  llvm::DenseMap<TypeID, llvm::SmallVector<Pattern *>> traitPatternMap;
  llvm::DenseMap<TypeID, llvm::SmallVector<Pattern *>> interfacePatternMap;
};

utils::LogicalResult applyPatternConversion(Module *module,
                                            const PatternSet &patterns);

} // namespace kecc::ir

#endif // KECC_IR_PATTERN_MATCH_H
