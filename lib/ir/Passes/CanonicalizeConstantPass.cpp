#include "kecc/ir/IRAttributes.h"
#include "kecc/ir/IRInstructions.h"
#include "kecc/ir/IRTransforms.h"
#include "kecc/ir/Instruction.h"

namespace kecc::ir {

PassResult CanonicalizeConstant::run(Module *module) {
  auto constantBlock = module->getIR()->getConstantBlock();

  llvm::DenseMap<ConstantAttr, InstructionStorage *> constantMap;
  llvm::DenseMap<InstructionStorage *, InstructionStorage *> replaceMap;

  for (auto iter = constantBlock->begin(); iter != constantBlock->end();
       ++iter) {
    auto inst = *iter;
    auto constInst = inst->getDefiningInst<inst::Constant>();
    auto constAttr = constInst.getValue();

    if (!maintainStringFloat) {
      if (auto sfloatAttr = constAttr.dyn_cast<ConstantStringFloatAttr>()) {
        auto floatAttr = sfloatAttr.convertToFloatAttr();
        constInst.replaceValue(floatAttr);
      }
    }

    auto it = constantMap.find(constAttr);
    if (it != constantMap.end()) {
      auto *existingInst = it->second;
      if (existingInst != inst) {
        auto [_, inserted] = replaceMap.try_emplace(inst, existingInst);
        assert(inserted && "Instruction should not be replaced multiple times");
        (void)inserted;
      }
    } else {
      constantMap.try_emplace(constAttr, inst);
    }
  }

  for (const auto &[from, to] : replaceMap) {
    module->replaceInst(from, to->getResults(), true);
  }

  return PassResult::success();
}

} // namespace kecc::ir
