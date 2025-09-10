#ifndef KECC_IR_TRANSFORMATIONS_H
#define KECC_IR_TRANSFORMATIONS_H

#include "kecc/ir/Pass.h"
namespace kecc::ir {

class CanonicalizeConstant final : public Pass {
public:
  PassResult run(Module *module) override;

  static llvm::StringRef getPassName() { return "canonicalize-constant"; }

  llvm::StringRef getPassArgument() const override {
    return "canonicalize-constant";
  }
  llvm::StringRef getDescription() const override {
    return "Canonicalize constant values in the program.";
  }

  void setOption(llvm::StringRef option) override {
    maintainStringFloat = (option == "maintain-string-float");
  }

private:
  bool maintainStringFloat = true;
};

class SimplifyCFG : public Pass {
public:
  PassResult run(Module *module) override;
  static llvm::StringRef getPassName() { return "simplify-cfg"; }
  llvm::StringRef getPassArgument() const override { return getPassName(); }
  llvm::StringRef getDescription() const override {
    return "Simplify the control flow graph by removing unnecessary branches, "
           "merging blocks, and eliminating unreachable code.";
  }
};

void registerSimplifyCFGPass();

class Mem2RegImpl;
class Mem2Reg : public FuncPass {
public:
  Mem2Reg();
  ~Mem2Reg();

  void init(Module *module, Function *function) override;
  void exit(Module *module, Function *function) override;
  PassResult run(Module *module, Function *function) override;

  static llvm::StringRef getPassName() { return "mem2reg"; }
  llvm::StringRef getPassArgument() const override { return getPassName(); }
  llvm::StringRef getDescription() const override {
    return "Promote memory to register by eliminating memory accesses "
           "and replacing them with direct register accesses.";
  }

private:
  std::unique_ptr<Mem2RegImpl> impl;
};

class GVNImpl;
class GVN : public FuncPass {
public:
  GVN();
  ~GVN();

  void init(Module *module) override;
  void exit(Module *module) override;
  void init(Module *module, Function *function) override;
  PassResult run(Module *module, Function *function) override;

  static llvm::StringRef getPassName() { return "gvn"; }
  llvm::StringRef getPassArgument() const override { return getPassName(); }
  llvm::StringRef getDescription() const override {
    return "Global Value Numbering (GVN) to eliminate redundant computations "
           "by identifying equivalent expressions.";
  }

private:
  std::unique_ptr<GVNImpl> impl;
};

class DeadCode : public Pass {
public:
  PassResult run(Module *module) override;

  static llvm::StringRef getPassName() { return "dead-code-elimination"; }
  llvm::StringRef getPassArgument() const override { return getPassName(); }
  llvm::StringRef getDescription() const override {
    return "Remove dead code that does not affect the program's output.";
  }
};

class OutlineConstantPass : public Pass {
public:
  void init(Module *module) override;
  PassResult run(Module *module) override;

  static llvm::StringRef getPassName() { return "outline-constant"; }
  llvm::StringRef getPassArgument() const override { return getPassName(); }
  llvm::StringRef getDescription() const override {
    return "Outline constant expressions for allocating them to registers, "
           "strictly";
  }
};

class InstructionFold : public Pass {
public:
  PassResult run(Module *module) override;

  static llvm::StringRef getPassName() { return "constant-fold"; }
  llvm::StringRef getPassArgument() const override { return getPassName(); }
  llvm::StringRef getDescription() const override {
    return "Fold constant expressions to their evaluated values.";
  }
};

class OutlineMultipleResults : public Pass {
public:
  PassResult run(Module *module) override;

  static llvm::StringRef getPassName() { return "outline-multiple-results"; }
  llvm::StringRef getPassArgument() const override { return getPassName(); }
  llvm::StringRef getDescription() const override {
    return "Outline multiple results into a single or couple of results.";
  }
};

class CanonicalizeStructImpl;
class CanonicalizeStruct : public Pass {
public:
  CanonicalizeStruct();
  ~CanonicalizeStruct();
  void init(Module *module) override;
  void exit(Module *module) override;
  PassResult run(Module *module) override;

  static llvm::StringRef getPassName() { return "canonicalize-struct"; }
  llvm::StringRef getPassArgument() const override { return getPassName(); }
  llvm::StringRef getDescription() const override {
    return "Canonicalize struct types by converting them to a more "
           "standardized form, such as flattening nested structs.";
  }

private:
  void convertAllFuncT(Module *module);

  std::unique_ptr<CanonicalizeStructImpl> impl;
};

void registerCanonicalizeStructPasses();

class InlineCallPass : public Pass {
public:
  PassResult run(Module *module) override;

  static llvm::StringRef getPassName() { return "inline-call"; }
  llvm::StringRef getPassArgument() const override { return getPassName(); }
  llvm::StringRef getDescription() const override {
    return "Inline function call instruction";
  }
};

class CreateFunctionArgument : public FuncPass {
public:
  PassResult run(Module *module, Function *func) override;

  static llvm::StringRef getPassName() { return "create-function-argument"; }
  llvm::StringRef getPassArgument() const override { return getPassName(); }
  llvm::StringRef getDescription() const override {
    return "Create function argument instructions for all functions in the "
           "module.";
  }
};

class ConversionToCopyPass : public Pass {
public:
  void init(Module *module) override;
  PassResult run(Module *module) override;

  static llvm::StringRef getPassName() { return "conversion-to-copy"; }
  llvm::StringRef getPassArgument() const override { return getPassName(); }
  llvm::StringRef getDescription() const override {
    return "Conversion operations to copy operations, if possible";
  }
};

} // namespace kecc::ir

#endif // KECC_IR_TRANSFORMATIONS_H
