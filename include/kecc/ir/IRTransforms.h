#ifndef KECC_IR_TRANSFORMATIONS_H
#define KECC_IR_TRANSFORMATIONS_H

#include "kecc/ir/Pass.h"
namespace kecc::ir {

class CanonicalizeConstant : public Pass {
public:
  PassResult run(Module *module) override;

  llvm::StringRef getPassName() const override {
    return "canonicalize-constant";
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
  llvm::StringRef getPassName() const override { return "simplify-cfg"; }
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

  llvm::StringRef getPassName() const override { return "mem2reg"; }

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

  llvm::StringRef getPassName() const override { return "gvn"; }

private:
  std::unique_ptr<GVNImpl> impl;
};

class DeadCode : public Pass {
public:
  PassResult run(Module *module) override;

  llvm::StringRef getPassName() const override {
    return "dead-code-elimination";
  }
};

class OutlineConstantPass : public Pass {
public:
  void init(Module *module) override;
  PassResult run(Module *module) override;

  llvm::StringRef getPassName() const override { return "outline-constant"; }
};

class InstructionFold : public Pass {
public:
  PassResult run(Module *module) override;

  llvm::StringRef getPassName() const override { return "constant-fold"; }
};

class OutlineMultipleResults : public Pass {
public:
  PassResult run(Module *module) override;

  llvm::StringRef getPassName() const override {
    return "outline-multiple-results";
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

  llvm::StringRef getPassName() const override { return "canonicalize-struct"; }

private:
  void convertAllFuncT(Module *module);

  std::unique_ptr<CanonicalizeStructImpl> impl;
};

void registerCanonicalizeStructPasses();

class InlineCallPass : public Pass {
public:
  PassResult run(Module *module) override;

  llvm::StringRef getPassName() const override { return "inline-call"; }
};

} // namespace kecc::ir

#endif // KECC_IR_TRANSFORMATIONS_H
