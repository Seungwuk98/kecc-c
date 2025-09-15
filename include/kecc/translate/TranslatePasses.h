#ifndef KECC_TRANSLATE_PASSES_H
#define KECC_TRANSLATE_PASSES_H

#include "kecc/ir/Module.h"
#include "kecc/ir/Pass.h"

namespace kecc::translate {

class ConversionToCopyPass : public ir::Pass {
public:
  void init(ir::Module *module) override;
  ir::PassResult run(ir::Module *module) override;

  static llvm::StringRef getPassName() { return "conversion-to-copy"; }
  llvm::StringRef getPassArgument() const override { return getPassName(); }
  llvm::StringRef getDescription() const override {
    return "Conversion operations to copy operations, if possible";
  }
};

class InlineMemoryInstPass : public ir::Pass {
public:
  void init(ir::Module *module) override;

  ir::PassResult run(ir::Module *module) override;

  static llvm::StringRef getPassName() { return "inline-memory-inst"; }

  llvm::StringRef getPassArgument() const override { return getPassName(); }
  llvm::StringRef getDescription() const override {
    return "Inline memory instructions, if possible";
  }
};

void addO1Passes(ir::PassManager &pm);

} // namespace kecc::translate

#endif // KECC_TRANSLATE_PASSES_H
