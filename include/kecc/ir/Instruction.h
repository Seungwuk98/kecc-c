#ifndef KECC_IR_INSTRUCTIONS_H
#define KECC_IR_INSTRUCTIONS_H

#include "kecc/ir/Attribute.h"
#include "kecc/ir/Interface.h"
#include "kecc/ir/Type.h"
#include "kecc/ir/Value.h"
#include "kecc/ir/WalkSupport.h"
#include "kecc/utils/List.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/TrailingObjects.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <type_traits>

namespace kecc::ir {

class Block;
class InstructionStorage;
class IRPrintContext;
class AbstractInstruction;
class JumpArg;

class Instruction {
public:
  Instruction() : impl(nullptr) {}
  Instruction(const InstructionStorage *impl)
      : impl(const_cast<InstructionStorage *>(impl)) {}

  bool operator==(const Instruction &other) const { return impl == other.impl; }
  bool operator!=(const Instruction &other) const { return impl != other.impl; }
  operator bool() const { return impl != nullptr; }
  InstructionStorage *operator->() const {
    assert(impl && "InstructionStorage is null");
    return impl;
  }

  template <typename... U> bool isa() const { return llvm::isa<U...>(*this); }
  template <typename U> U cast() const { return llvm::cast<U>(*this); }
  template <typename U> U dyn_cast() const { return llvm::dyn_cast<U>(*this); }
  template <typename U> U dyn_cast_or_null() const {
    return llvm::dyn_cast_or_null<U>(*this);
  }

  TypeID getId() const;

  llvm::SMRange getRange() const;

  InstructionStorage *getStorage() const { return impl; }

  IRContext *getContext() const;

  void print(IRPrintContext &context) const;
  void print(llvm::raw_ostream &os) const;
  void dump() const;

  template <typename Interface> bool implementInterface() const;
  bool implementInterface(TypeID interfaceId) const;

  template <template <typename> typename Trait> bool hasTrait() const;
  bool hasTrait(TypeID traitId) const;

  Block *getParentBlock() const;

private:
  InstructionStorage *impl;
};

template <typename ConcreteType, typename Traits>
class InstructionInterface
    : public Interface<ConcreteType, Instruction, Traits> {
public:
  static bool classof(Instruction inst) {
    return inst.implementInterface<ConcreteType>();
  }

private:
};

class JumpArgState {
public:
  JumpArgState() = default;
  JumpArgState(Block *block) : block(block) {}
  JumpArgState(Block *block, llvm::ArrayRef<Value> args)
      : block(block), args(args.begin(), args.end()) {}
  JumpArgState(Block *block, Value arg) : block(block), args(1, arg) {}

  Block *getBlock() const { return block; }
  llvm::ArrayRef<Value> getArgs() const { return args; }

  void setBlock(Block *newBlock) { block = newBlock; }
  void setArgs(llvm::ArrayRef<Value> newArgs) {
    args.assign(newArgs.begin(), newArgs.end());
  }
  void setArg(std::size_t index, Value arg) { args[index] = arg; }
  void pushArg(Value arg) { args.emplace_back(arg); }

private:
  friend class JumpArg;
  Block *block;
  llvm::SmallVector<Value> args;
};

class InstructionState {
public:
  InstructionState() = default;
  InstructionState(const InstructionState &) = default;
  InstructionState(InstructionState &&) = default;
  InstructionState &operator=(const InstructionState &) = default;
  InstructionState &operator=(InstructionState &&) = default;

  void setRange(llvm::SMRange range) { this->range = range; }
  void setOperands(llvm::ArrayRef<Value> operands);
  void pushOperand(Value operand);
  void setTypes(llvm::ArrayRef<Type> types);
  void pushType(Type type);
  void pushType(Type type, llvm::StringRef name);
  void setAttributes(llvm::ArrayRef<Attribute> names);
  void pushAttribute(Attribute name);
  void setParentBlock(Block *block) { this->parentBlock = block; }
  void setJumpArgs(llvm::ArrayRef<JumpArgState> jumpArgs);
  void pushJumpArg(const JumpArgState &jumpArg);

private:
  friend class InstructionStorage;

  llvm::SMRange range;
  llvm::SmallVector<Value> operands;
  llvm::SmallVector<std::pair<Type, llvm::StringRef>> types;
  llvm::SmallVector<Attribute> attributes;
  llvm::SmallVector<JumpArgState> jumpArgs;
  Block *parentBlock;
};

class InstructionStorage final
    : public llvm::TrailingObjects<InstructionStorage, Operand, Attribute,
                                   JumpArg *> {
public:
  size_t numTrailingObjects(OverloadToken<Operand>) const {
    return operandSize;
  }
  size_t numTrailingObjects(OverloadToken<Attribute>) const {
    return attributeSize;
  }
  size_t numTrailingObjects(OverloadToken<JumpArg *>) const {
    return jumpArgSize;
  }

  llvm::SMRange getRange() const;
  llvm::ArrayRef<Operand> getOperands() const;
  llvm::ArrayRef<Attribute> getAttributes() const;
  llvm::ArrayRef<JumpArg *> getJumpArgs() const;

  static InstructionStorage *create(InstructionState &state);

  void destroy();

  Value getResult(std::uint8_t valueNumber) const;
  std::uint8_t getResultSize() const { return resultSize; }
  // Returns the results of this instruction as a vector.
  // Result memory is not contiguous, so this is a copy.
  llvm::SmallVector<Value> getResults() const;

  const Operand &getOperand(std::size_t index) const;
  std::size_t getOperandSize() const { return operandSize; }
  Attribute getAttribute(std::size_t index) const;
  std::size_t getAttributeSize() const { return attributeSize; }
  JumpArg *getJumpArg(std::size_t index) const;
  std::size_t getJumpArgSize() const { return jumpArgSize; }

  bool hasInterface(TypeID interfaceId) const;

  WalkResult walk(llvm::function_ref<WalkResult(const Operand &)> callback);

  AbstractInstruction *getAbstractInstruction() const { return abstractInst; }
  void setAbstractInstruction(AbstractInstruction *abstractInst) {
    this->abstractInst = abstractInst;
  }

  template <typename Inst> Inst getDefiningInst();

  void print(IRPrintContext &context) const;
  void print(llvm::raw_ostream &os) const;
  void dump() const;

  void replaceOperand(Value oldV, Value newV);
  Block *getParentBlock() const { return parentBlock; }

  void setRange(llvm::SMRange range) { this->range = range; }
  void setAttribute(std::size_t index, Attribute attr);
  void setAttributes(llvm::ArrayRef<Attribute> attrs);
  void setOperand(std::size_t index, Value operand);
  void setOperands(llvm::ArrayRef<Value> operands);
  void setJumpArg(std::size_t index, JumpArgState jumpArg);
  void setJumpArgs(llvm::ArrayRef<JumpArgState> jumpArgs);
  void setParentBlock(Block *block) { parentBlock = block; }

  IRContext *getContext() const;

  void dropReferences();

  template <typename Interface> bool implementInterface() const {
    return implementInterface(TypeID::get<Interface>());
  }
  bool implementInterface(TypeID interfaceId) const;

  template <template <typename> typename Trait> bool hasTrait() const {
    return hasTrait(TypeID::get<Trait>());
  }
  bool hasTrait(TypeID traitId) const;

private:
  friend class IRBuilder;
  InstructionStorage(std::uint8_t resultSize, std::size_t operandSize,
                     std::size_t attributeSize, std::size_t jumpArgSize,
                     llvm::SMRange range, Block *parentBlock);

  ValueImpl *getResultMemory(std::uint8_t valueNumber);

  std::size_t getPrefixBytes() const;

  const std::uint8_t resultSize;
  const std::size_t operandSize;
  const std::size_t attributeSize;
  const std::size_t jumpArgSize;

  llvm::SMRange range;
  Block *parentBlock;
  AbstractInstruction *abstractInst;
};

template <typename Interface> bool Instruction::implementInterface() const {
  return impl && impl->hasInterface(TypeID::get<Interface>());
}

class AbstractInstruction {
public:
  using PrintFn = std::function<void(Instruction, IRPrintContext &)>;
  using HasTraitFn = std::function<bool(TypeID)>;

  TypeID getId() const { return id; }

  IRContext *getContext() const { return context; }

  template <typename Interface>
  typename Interface::Concept *getConcept() const {
    return interfaces.lookup<Interface>();
  }

  bool hasInterface(TypeID interfaceId) const {
    return interfaces.contains(interfaceId);
  }

  template <typename ConcreteInst>
  static AbstractInstruction build(IRContext *context) {
    TypeID id = TypeID::get<ConcreteInst>();
    InterfaceMap interfaces = ConcreteInst::buildInterfaceMap(context);
    PrintFn printFn = ConcreteInst::getPrintFn();
    HasTraitFn hasTraitFn = ConcreteInst::getHasTraitFn();

    return AbstractInstruction(id, context, std::move(interfaces),
                               std::move(printFn), std::move(hasTraitFn));
  }

  const auto &getPrintFn() const { return printFn; }
  const auto &getHasTraitFn() const { return hasTraitFn; }

private:
  AbstractInstruction(TypeID id, IRContext *context, InterfaceMap interfaces,
                      PrintFn printFn, HasTraitFn traitFn)
      : id(id), context(context), interfaces(std::move(interfaces)),
        printFn(std::move(printFn)), hasTraitFn(std::move(traitFn)) {}

  TypeID id;
  IRContext *context;
  InterfaceMap interfaces;
  PrintFn printFn;
  HasTraitFn hasTraitFn;
};

template <typename Inst> Inst InstructionStorage::getDefiningInst() {
  return Instruction(this).dyn_cast<Inst>();
}

template <template <typename> typename Trait>
bool Instruction::hasTrait() const {
  TypeID traitId = TypeID::get<Trait>();
  return impl->getAbstractInstruction()->getHasTraitFn()(traitId);
}

template <typename Trait>
concept HasVerifyFn = requires(InstructionStorage *storage) {
  { Trait::verifyTrait(storage) } -> std::convertible_to<bool>;
};

template <typename ConcreteInst, typename ParentInst,
          template <typename> typename... Trait>
class InstructionTemplate : public ParentInst, public Trait<ConcreteInst>... {
public:
  using ParentInst::getStorage;
  using ParentInst::ParentInst;
  using Base = InstructionTemplate<ConcreteInst, ParentInst, Trait...>;

  static InterfaceMap buildInterfaceMap(IRContext *context) {
    return InterfaceMap::build<Trait<ConcreteInst>...>(context);
  }

  template <typename U>
    requires std::is_base_of_v<Instruction, U>
  static bool classof(U inst) {
    return inst.getId() == TypeID::get<ConcreteInst>();
  }

  static auto getHasTraitFn() {
    static TypeID traitIds[] = {
        TypeID::get<Trait>()...,
    };
    return [](TypeID traitID) -> bool {
      for (auto idx = 0u; idx < sizeof...(Trait); ++idx) {
        if (traitIds[idx] == traitID)
          return true;
      }
      return false;
    };
  }

  static auto getPrintFn() {
    return [](Instruction inst, IRPrintContext &context) {
      ConcreteInst::printer(inst.cast<ConcreteInst>(), context);
    };
  }

  static bool verifyTrait(InstructionStorage *storage) {
    return (verifyTraitImpl<Trait<ConcreteInst>>(storage) && ...);
  }

private:
  template <typename T>
  static bool verifyTraitImpl(InstructionStorage *storage) {
    if constexpr (HasVerifyFn<T>) {
      return T::verifyTrait(storage);
    } else {
      return true; // No verification needed for this trait
    }
  }
};

class ValuePrinter {
public:
  virtual ~ValuePrinter() = default;
  virtual void printValue(Value value, IRPrintContext &context,
                          bool printName) = 0;
  virtual void printOperand(const Operand &operand,
                            IRPrintContext &context) = 0;
  virtual void printJumpArg(const JumpArg *jumpArg,
                            IRPrintContext &context) = 0;
};

class DefaultValuePrinter : public ValuePrinter {
public:
  DefaultValuePrinter(IRPrintContext &context) : context(context) {}

  void printValue(Value value, IRPrintContext &context,
                  bool printName) override;

  void printOperand(const Operand &operand, IRPrintContext &context) override;

  void printJumpArg(const JumpArg *jumpArg, IRPrintContext &context) override;

private:
  IRPrintContext &context;
};

class IRPrintContext {
public:
  enum PrintMode {
    Default,
    Debug,
  };

  IRPrintContext(llvm::raw_ostream &os, PrintMode mode = Default)
      : mode(mode), os(os) {
    valuePrinter = std::make_unique<DefaultValuePrinter>(*this);
  }
  IRPrintContext(llvm::raw_ostream &os, std::unique_ptr<ValuePrinter> printer,
                 PrintMode mode = Default)
      : mode(mode), os(os), valuePrinter(std::move(printer)) {}

  struct AddIndent {
    AddIndent(IRPrintContext &context) : context(context) { context.indent++; }
    ~AddIndent() { context.indent--; }

  private:
    IRPrintContext &context;
  };

  RegisterId getId(Value value) const;
  void clearIdMap();
  void setId(Value value, RegisterId id);
  void setMode(PrintMode newMode) { mode = newMode; }
  bool isDebugMode() const { return mode == PrintMode::Debug; }

  void printIndent();

  llvm::raw_ostream &getOS() { return os; }

  void setIndentWidth(unsigned width) { indentWidth = width; }

  void printValue(Value value, bool printName = false) {
    valuePrinter->printValue(value, *this, printName);
  }

  void printValues(llvm::ArrayRef<Value> values, bool printName = false);

  void printOperand(const Operand &operand) {
    valuePrinter->printOperand(operand, *this);
  }

  void printJumpArg(const JumpArg *jumpArg) {
    valuePrinter->printJumpArg(jumpArg, *this);
  }

private:
  unsigned indent = 0;
  unsigned indentWidth = 2;
  PrintMode mode;
  llvm::raw_ostream &os;
  std::unique_ptr<ValuePrinter> valuePrinter;
  llvm::DenseMap<Value, RegisterId> idMap;
};

template <typename ConcreteType, template <typename> typename ConcreteTrait>
struct TraitBase {
protected:
  InstructionStorage *getStorage() const {
    return static_cast<const ConcreteType *>(this)->getStorage();
  }
};

template <size_t N> struct NOperand {
  static_assert(N > 1, "Must use 'OneResult' or 'ZeroResult'");
  template <typename ConcreteType>
  struct Trait : TraitBase<ConcreteType, Trait> {
    static bool verifyTrait(InstructionStorage *impl) {
      return impl->getResultSize() == N;
    }
  };
};

template <typename ConcreteType>
struct Terminator : TraitBase<ConcreteType, Terminator> {};

template <typename ConcreteType> struct Arg : TraitBase<ConcreteType, Arg> {};

template <typename ConcreteType>
struct OneResult : TraitBase<ConcreteType, OneResult> {

  operator Value() const { return this->getStorage()->getResult(0); }

  Type getType() const { return Value(*this).getType(); }
  Value getResult() const { return Value(*this); }
  void printAsOperand(IRPrintContext &context, bool printName = false) const {
    getResult().printAsOperand(context, printName);
  }
  void setValueName(llvm::StringRef name) {
    getResult().getImpl()->setValueName(name);
  }
  llvm::StringRef getValueName() const {
    return getResult().getImpl()->getValueName();
  }

  static bool verifyTrait(InstructionStorage *impl) {
    return impl->getResultSize() == 1;
  }
};

template <typename ConcreteType>
struct ZeroResult : TraitBase<ConcreteType, ZeroResult> {
  static bool verifyTrait(InstructionStorage *impl) {
    return impl->getResultSize() == 0;
  }
};

template <typename ConcreteType>
struct VariadicResults : TraitBase<ConcreteType, VariadicResults> {

  std::size_t getNumResults() const {
    return this->getStorage()->getResultSize();
  }

  Value getResult(std::size_t idx) const {
    return this->getStorage()->getResult(idx);
  }
};

template <typename ConcreteType>
struct SideEffect : TraitBase<ConcreteType, SideEffect> {};

template <typename ConcreteType>
struct ReadMemory : TraitBase<ConcreteType, ReadMemory> {};

template <typename ConcreteType>
struct WriteMemory : TraitBase<ConcreteType, WriteMemory> {};

template <typename ConcreteType> struct Pure : TraitBase<ConcreteType, Pure> {};

template <typename ConcreteType>
struct CallLike : TraitBase<ConcreteType, CallLike> {};

class IRBuilder;

class Phi : public InstructionTemplate<Phi, Instruction, OneResult, Arg> {
public:
  using Base::Base;
  static void build(IRBuilder &builder, InstructionState &state, Type type);

  static void printer(Phi op, IRPrintContext &context);
};

class BlockExit : public Instruction {
public:
  using Instruction::Instruction;

  static bool classof(Instruction inst);

  WalkResult walk(llvm::function_ref<WalkResult(JumpArg *)> callback);
};

} // namespace kecc::ir

namespace llvm {

template <> struct DenseMapInfo<kecc::ir::Instruction> {
  using Instruction = kecc::ir::Instruction;

  static inline Instruction getEmptyKey() { return Instruction(nullptr); }

  static inline Instruction getTombstoneKey() {
    return DenseMapInfo<kecc::ir::InstructionStorage *>::getTombstoneKey();
  }

  static unsigned getHashValue(const Instruction &inst) {
    return DenseMapInfo<kecc::ir::InstructionStorage *>::getHashValue(
        inst.getStorage());
  }

  static bool isEqual(const Instruction &LHS, const Instruction &RHS) {
    return LHS == RHS;
  }
};

template <typename T>
struct DenseMapInfo<
    T, std::enable_if_t<std::is_base_of_v<kecc::ir::Instruction, T>>>
    : DenseMapInfo<kecc::ir::Instruction> {
  static inline T getEmptyKey() { return nullptr; }
  static inline T getTombstoneKey() {
    return DenseMapInfo<kecc::ir::InstructionStorage *>::getTombstoneKey();
  }
};

template <typename To, typename From>
struct CastInfo<
    To, From,
    std::enable_if_t<
        std::is_same_v<kecc::ir::Instruction, std::remove_const_t<From>> ||
        std::is_base_of_v<kecc::ir::Instruction, From>>>
    : NullableValueCastFailed<To>,
      DefaultDoCastIfPossible<To, From, CastInfo<To, From>> {
  static inline bool isPossible(kecc::ir::Instruction inst) {
    if constexpr (std::is_base_of_v<To, From>) {
      return true;
    } else {
      return To::classof(inst);
    }
  }
  static inline To doCast(kecc::ir::Instruction inst) {
    return To(inst.getStorage());
  }
};

template <typename Inst>
struct CastInfo<Inst, kecc::ir::InstructionStorage *>
    : ValueFromPointerCast<Inst, kecc::ir::InstructionStorage,
                           CastInfo<Inst, kecc::ir::InstructionStorage *>> {
  static inline bool isPossible(kecc::ir::InstructionStorage *storage) {
    return Inst::classof(kecc::ir::Instruction(storage));
  }
};

template <typename Inst>
struct CastInfo<Inst, const kecc::ir::InstructionStorage *>
    : ConstStrippingForwardingCast<
          Inst, const kecc::ir::InstructionStorage *,
          CastInfo<Inst, kecc::ir::InstructionStorage *>> {};

} // namespace llvm

#endif // KECC_IR_INSTRUCTIONS_H
