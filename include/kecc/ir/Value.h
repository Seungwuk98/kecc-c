#ifndef KECC_IR_OPERAND_H
#define KECC_IR_OPERAND_H

#include "kecc/ir/Type.h"
#include "kecc/utils/List.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/Support/raw_ostream.h"
#include <ostream>

namespace kecc::ir {

class IRPrintContext;
class InstructionStorage;
class Operand;

class ValueImpl : public utils::ListObject<ValueImpl, Operand *> {
public:
  using UsageNode = utils::ListObject<ValueImpl, Operand *>::Node;

  ~ValueImpl() {
    assert(empty() && "Usage list must be empty before deletion");
  }

  ValueImpl(llvm::StringRef name, std::uint8_t valueNumber, Type type)
      : valueName(name), valueNumber(valueNumber), type(type) {}

  static void deleteObject(Operand *storage) {
    // Do not delete operand.
    // It is just a usage reference.
  }

  InstructionStorage *getInstruction() const;

  Type getType() const { return type; }
  void setType(Type newType) { type = newType; }

  std::uint8_t getValueNumber() const { return valueNumber; }

  llvm::StringRef getValueName() const { return valueName; }
  void setValueName(llvm::StringRef name) { valueName = name; }

private:
  std::string valueName;
  std::uint8_t valueNumber;
  Type type;
};

class Value {
public:
  Value() : impl(nullptr) {}
  Value(ValueImpl *impl) : impl(impl) {}
  Value(const Value &other) : impl(other.impl) {}

  InstructionStorage *getInstruction() const;

  template <typename Inst> Inst getDefiningInst() const {
    return llvm::dyn_cast<Inst>(getInstruction());
  }

  operator bool() const { return impl != nullptr; }
  bool operator==(const Value &other) const { return impl == other.impl; }
  bool operator!=(const Value &other) const { return impl != other.impl; }

  ValueImpl *getImpl() const { return impl; }

  void printAsOperand(IRPrintContext &context, bool printName = false) const;
  void printAsOperand(llvm::raw_ostream &os, bool printName = false) const;

  Type getType() const { return getImpl()->getType().constCanonicalize(); }

  friend inline llvm::hash_code hash_value(Value value) {
    return llvm::DenseMapInfo<ValueImpl *>::getHashValue(value.getImpl());
  }

  void replaceWith(Value newValue);

  auto useBegin() const { return impl->begin(); }
  auto useEnd() const { return impl->end(); }

  bool hasUses() const { return !impl->empty(); }

  llvm::StringRef getValueName() const { return impl->getValueName(); }

  bool isConstant() const;

protected:
  ValueImpl *impl;
};

class ValueArray {
public:
  ValueArray(llvm::ArrayRef<ValueImpl *> values) : values(values) {}

  Value operator[](size_t index) const { return Value(values[index]); }

  size_t size() const { return values.size(); }

  bool empty() const { return values.empty(); }

  struct Iterator {
    Iterator(llvm::ArrayRef<ValueImpl *>::iterator it) : it(it) {}
    Iterator &operator++() {
      ++it;
      return *this;
    }
    Iterator operator++(int) {
      Iterator tmp = *this;
      ++it;
      return tmp;
    }
    Iterator &operator--() {
      --it;
      return *this;
    }
    Iterator operator--(int) {
      Iterator tmp = *this;
      --it;
      return tmp;
    }
    bool operator==(const Iterator &other) const { return it == other.it; }
    bool operator!=(const Iterator &other) const { return it != other.it; }
    Value operator*() const { return Value(*it); }
    ValueImpl *operator->() const { return *it; }

    llvm::ArrayRef<ValueImpl *>::iterator it;
  };

  Iterator begin() const { return Iterator(values.begin()); }
  Iterator end() const { return Iterator(values.end()); }

private:
  llvm::ArrayRef<ValueImpl *> values;
};

class Operand : public Value {
public:
  using Value::Value;
  Operand(InstructionStorage *storage, Value value);
  ~Operand();

  Operand(const Operand &) = delete;
  Operand &operator=(const Operand &) = delete;
  Operand(Operand &&other) noexcept { *this = std::move(other); }
  Operand &operator=(Operand &&other) noexcept;

  void insertUsage(ValueImpl *usageList);

  void drop();

  InstructionStorage *getOwner() const { return owner; }

private:
  ValueImpl::UsageNode *usageNode;
  InstructionStorage *owner;
};

} // namespace kecc::ir

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const kecc::ir::Value &value) {
  value.printAsOperand(os);
  return os;
}

inline std::ostream &operator<<(std::ostream &os,
                                const kecc::ir::Value &value) {
  std::string result;
  llvm::raw_string_ostream osStream(result);
  value.printAsOperand(osStream);
  return os << result;
}

namespace llvm {

template <> struct DenseMapInfo<kecc::ir::Value> {

  static kecc::ir::Value getEmptyKey() { return kecc::ir::Value(nullptr); }

  static kecc::ir::Value getTombstoneKey() {
    return llvm::DenseMapInfo<kecc::ir::ValueImpl *>::getTombstoneKey();
  }

  static unsigned getHashValue(kecc::ir::Value value) {
    return DenseMapInfo<kecc::ir::ValueImpl *>::getHashValue(value.getImpl());
  }

  static bool isEqual(kecc::ir::Value lhs, kecc::ir::Value rhs) {
    return lhs == rhs;
  }
};

template <typename T>
struct DenseMapInfo<T, std::enable_if_t<std::is_base_of_v<kecc::ir::Value, T>>>
    : public DenseMapInfo<kecc::ir::Value> {
  static T getEmptyKey() { return T(nullptr); }

  static T getTombstoneKey() {
    return DenseMapInfo<kecc::ir::ValueImpl *>::getTombstoneKey();
  }
};

} // namespace llvm

#endif // KECC_IR_OPERAND_H
