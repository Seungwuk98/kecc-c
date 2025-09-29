#ifndef KECC_IR_INTERPRETER_H
#define KECC_IR_INTERPRETER_H

#include "kecc/ir/IR.h"
#include "kecc/ir/IRAnalyses.h"
#include "kecc/ir/IRInstructions.h"
#include "kecc/ir/Module.h"
#include "kecc/ir/Value.h"
#include "kecc/utils/PointerCastBase.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/MemAlloc.h"
#include "llvm/Support/raw_ostream.h"
#include <variant>

namespace kecc::ir {

class VRegister : public utils::PointerCastBase<VRegister> {
public:
  enum Kind {
    Int,
    Float,
    Struct,
  };

  VRegister(Kind kind) : kind(kind) {}

  virtual ~VRegister() = default;

  Kind getKind() const { return kind; }

  void dump() const;
  virtual void print(llvm::raw_ostream &os, Type type,
                     StructSizeAnalysis *analysis) const = 0;
  virtual void mv(VRegister *src) = 0;
  virtual std::unique_ptr<VRegister> clone() const = 0;

  static llvm::StringRef kindToString(Kind kind);

private:
  Kind kind;
};

class VMemory {
public:
  VMemory() : data(nullptr) {}
  VMemory(void *data) : data(data) {}

  void *getData() const { return data; }
  VMemory getElementPtr(int offset) const;

  void loadInto(VRegister *dest, Type type, StructSizeAnalysis *analysis) const;
  void storeFrom(VRegister *src, Type type, StructSizeAnalysis *analysis);

  void print(llvm::raw_ostream &os, Type type,
             StructSizeAnalysis *analysis) const;

private:
  void *data;
};

class VRegisterInt : public VRegister {
public:
  VRegisterInt() : VRegister(Kind::Int), value(0) {}
  VRegisterInt(std::uint64_t v) : VRegister(Kind::Int), value(v) {}

  std::uint64_t getValue() const { return value; }
  void setValue(std::uint64_t v) { value = v; }
  void setValue(llvm::APSInt v);
  void setValue(VMemory v);

  llvm::APSInt getAsInteger(unsigned bitWidth, bool isSigned) const;
  VMemory getAsMemory() const;

  static bool classof(const VRegister *v) { return v->getKind() == Kind::Int; }
  void print(llvm::raw_ostream &os, Type type,
             StructSizeAnalysis *analysis) const override;
  void mv(VRegister *src) override;
  std::unique_ptr<VRegister> clone() const override;

private:
  static_assert(sizeof(VMemory) == sizeof(std::uint64_t));
  union {
    std::uint64_t value;
    VMemory memory;
  };
};

class VRegisterFloat : public VRegister {
public:
  VRegisterFloat() : VRegister(Kind::Float), value(0) {}
  VRegisterFloat(std::uint64_t v) : VRegister(Kind::Float), value(v) {}

  std::uint64_t getValue() const { return value; }
  void setValue(llvm::APFloat v);
  void setValue(std::uint64_t v);

  llvm::APFloat getAsFloat(int bitwidth) const;
  void print(llvm::raw_ostream &os, Type type,
             StructSizeAnalysis *analysis) const override;
  void mv(VRegister *src) override;
  std::unique_ptr<VRegister> clone() const override;

  static bool classof(const VRegister *v) {
    return v->getKind() == Kind::Float;
  }

private:
  std::uint64_t value;
};

class VRegisterDynamic : public VRegister {
public:
  VRegisterDynamic() : VRegister(Kind::Struct), size(0), data(nullptr) {}
  VRegisterDynamic(size_t size)
      : VRegister(Kind::Struct), size(size), data(llvm::safe_malloc(size)) {}
  ~VRegisterDynamic() {
    if (data)
      free(data);
  }

  void *getData() const { return data; }
  static bool classof(const VRegister *v) {
    return v->getKind() == Kind::Struct;
  }

  void print(llvm::raw_ostream &os, Type type,
             StructSizeAnalysis *analysis) const override;
  void mv(VRegister *src) override;
  std::unique_ptr<VRegister> clone() const override;

  size_t getSize() const { return size; }

private:
  size_t size;
  void *data;
};

class Interpreter;

class InterpValue {
public:
  InterpValue(VRegister *v) : value(v) {}
  InterpValue(std::unique_ptr<VRegister> v) : value(std::move(v)) {}

  VRegister *get() const {
    if (std::holds_alternative<VRegister *>(value)) {
      return std::get<VRegister *>(value);
    } else {
      return std::get<std::unique_ptr<VRegister>>(value).get();
    }
  }

  void print(llvm::raw_ostream &os, Type type,
             StructSizeAnalysis *analysis) const {
    get()->print(os, type, analysis);
  }

  auto operator->() const { return get(); }

private:
  std::variant<VRegister *, std::unique_ptr<VRegister>> value;
};

class GlobalTable {
public:
  GlobalTable() = default;

  void addGlobal(llvm::StringRef name, std::unique_ptr<VRegister> reg) {
    table[name] = std::move(reg);
  }

  VRegister *getGlobal(llvm::StringRef name) const {
    auto it = table.find(name);
    if (it != table.end()) {
      return it->second.get();
    }
    return nullptr;
  }

  llvm::BumpPtrAllocator &getAllocator() { return allocator; }

private:
  llvm::BumpPtrAllocator allocator;
  llvm::DenseMap<llvm::StringRef, std::unique_ptr<VRegister>> table;
};

class StackFrame {
public:
  StackFrame(Interpreter *interpreter, Function *function,
             GlobalTable *globalTable, llvm::SMRange callSite)
      : interpreter(interpreter), function(function), globalTable(globalTable),
        callSite(callSite) {
    initRegisters();
  }
  ~StackFrame();

  ir::Function *getFunction() const { return function; }

  std::unique_ptr<VRegister> constValue(inst::Constant c) const;

  InterpValue getRegister(ir::Value v) const {
    if (auto constant = v.getDefiningInst<inst::Constant>())
      return InterpValue(constValue(constant));

    auto it = registers.find(v);
    if (it != registers.end()) {
      return it->second;
    }
    llvm_unreachable("Register not found");
  }

  void print(IRPrintContext &context, bool summary = false) const;

  llvm::SmallVectorImpl<std::unique_ptr<VRegister>> &getReturnValues() {
    return returnValues;
  }

private:
  void initRegisters();

  Interpreter *interpreter;
  Function *function;
  GlobalTable *globalTable;
  llvm::SMRange callSite;
  void *frameMem = nullptr;
  size_t frameSize = 0;
  llvm::MapVector<ir::inst::LocalVariable, VMemory> localVars;
  llvm::MapVector<ir::Value, VRegister *> registers;
  llvm::SmallVector<std::unique_ptr<VRegister>> stackRegisters;
  llvm::SmallVector<std::unique_ptr<VRegister>> returnValues;
};

class Interpreter {
public:
  Interpreter(ir::Module *module) : module(module) {
    structSizeAnalysis =
        module->getOrCreateAnalysis<StructSizeAnalysis>(module);
    initGlobal();
  }

  struct CallStack {
    CallStack(Interpreter *interpreter, ir::Function *function)
        : interpreter(interpreter) {
      interpreter->callStack.emplace_back(
          std::unique_ptr<StackFrame>(new StackFrame(
              interpreter, function, &interpreter->globalTable, {})));
    }
    CallStack(Interpreter *interpreter, ir::Function *function,
              llvm::SMRange callSite)
        : interpreter(interpreter) {
      interpreter->callStack.emplace_back(
          std::unique_ptr<StackFrame>(new StackFrame(
              interpreter, function, &interpreter->globalTable, callSite)));
    }
    ~CallStack() { interpreter->callStack.pop_back(); }

  private:
    Interpreter *interpreter;
  };

  llvm::SmallVector<std::unique_ptr<VRegister>>
  call(llvm::StringRef name, llvm::ArrayRef<VRegister *> args,
       llvm::SMRange callSite = {});

  int callMain(llvm::ArrayRef<llvm::StringRef> args);

  StackFrame *getCurrentFrame() const {
    if (callStack.empty())
      return nullptr;
    return callStack.back().get();
  }
  StackFrame *getPreviousFrame() const {
    if (callStack.size() < 2)
      return nullptr;
    return callStack[callStack.size() - 2].get();
  }

  Module *getModule() const { return module; }

  void dumpAllStackFrames(llvm::raw_ostream &os = llvm::errs()) const;
  void dumpShortenedStackFrames(llvm::raw_ostream &os = llvm::errs()) const;

  struct PCGuard {
    PCGuard(Interpreter *interpreter)
        : interpreter(interpreter), oldPC(interpreter->programCounter) {}
    ~PCGuard() { interpreter->programCounter = oldPC; }

  private:
    Interpreter *interpreter;
    ir::InstructionStorage *oldPC;
  };

  void setPC(ir::InstructionStorage *pc) { programCounter = pc; }
  ir::InstructionStorage *getPC() const { return programCounter; }

  void setStackOverflowLimit(size_t limit) { stackOverflowLimit = limit; }

  StructSizeAnalysis *getStructSizeAnalysis() const {
    return structSizeAnalysis;
  }

private:
  void initGlobal();
  void printGlobalTable(llvm::raw_ostream &os) const;

  friend class StackFrame;
  llvm::SmallVector<std::unique_ptr<StackFrame>> callStack;
  GlobalTable globalTable;
  Module *module;
  StructSizeAnalysis *structSizeAnalysis;
  ir::InstructionStorage *programCounter = nullptr;
  size_t stackOverflowLimit = 1024;
};

} // namespace kecc::ir

#endif // KECC_IR_INTERPRETER_H
