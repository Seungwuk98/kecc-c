#include "kecc/ir/IRAnalyses.h"
#include "kecc/ir/IRInstructions.h"
#include "kecc/ir/IRTypes.h"

namespace kecc::ir {

struct StructSizeMapBuilder {
  StructSizeMapBuilder(Module *module) : module(module) {}

  std::pair<StructSizeMap, StructFieldsMap> calcStructSizeMap() const;

  Module *module;
  StructFieldsMap structFieldsMap;
  StructSizeMap structSizeMap;
};

std::unique_ptr<StructSizeAnalysis> StructSizeAnalysis::create(Module *module) {
  StructSizeMapBuilder builder(module);
  auto [sizeMap, fieldsMap] = builder.calcStructSizeMap();
  return std::unique_ptr<StructSizeAnalysis>(
      new StructSizeAnalysis(module, std::move(fieldsMap), std::move(sizeMap)));
}

static void
addStructToMap(StructSizeMap &sizeMap,
               const llvm::DenseMap<llvm::StringRef, llvm::SmallVector<Type>>
                   &structFieldsMap,
               llvm::StringRef name, llvm::ArrayRef<Type> fields) {
  if (sizeMap.contains(name))
    return;

  if (fields.empty()) {
    sizeMap.insert({name, {0, 1, {}}});
    return;
  }

  llvm::SmallVector<std::pair<size_t, size_t>> fieldsInfo;
  fieldsInfo.reserve(fields.size());
  for (Type fieldType : fields) {
    if (auto structType = fieldType.dyn_cast<NameStruct>()) {
      auto it = structFieldsMap.find(structType.getName());
      assert(it != structFieldsMap.end() &&
             "Struct type must be defined before it is used");
      addStructToMap(sizeMap, structFieldsMap, structType.getName(),
                     it->second);
    }
    auto [size, align] = fieldType.getSizeAndAlign(sizeMap);
    fieldsInfo.emplace_back(size, align);
  }

  size_t maxAlign = 0;
  llvm::for_each(fieldsInfo, [&](const auto &field) {
    maxAlign = std::max(maxAlign, field.second);
  });

  llvm::SmallVector<size_t> offsets;
  offsets.reserve(fieldsInfo.size());
  size_t currentOffset = 0;
  for (const auto &[fieldSize, fieldAlign] : fieldsInfo) {
    auto pad = currentOffset % fieldAlign
                   ? fieldAlign - (currentOffset % fieldAlign)
                   : 0;
    currentOffset += pad;
    offsets.emplace_back(currentOffset);
    currentOffset += fieldSize;
  }

  auto totalSize = ((currentOffset - 1) / maxAlign + 1) * maxAlign;
  sizeMap.insert({name, {totalSize, maxAlign, std::move(offsets)}});
}

std::pair<StructSizeMap, StructFieldsMap>
StructSizeMapBuilder::calcStructSizeMap() const {
  llvm::DenseMap<llvm::StringRef, llvm::SmallVector<Type>> structFieldsMap;

  for (InstructionStorage *inst : *module->getIR()->getStructBlock()) {
    inst::StructDefinition structDef =
        inst->getDefiningInst<inst::StructDefinition>();
    assert(structDef && "Instruction must be a struct definition");
    llvm::SmallVector<Type> fields;
    fields.reserve(structDef.getFieldSize());
    for (std::size_t i = 0; i < structDef.getFieldSize(); ++i) {
      fields.emplace_back(structDef.getField(i).second);
    }
    structFieldsMap.try_emplace(structDef.getName(), std::move(fields));
  }

  StructSizeMap structSizeMap;
  for (const auto &[name, fields] : structFieldsMap)
    addStructToMap(structSizeMap, structFieldsMap, name, fields);
  return {structSizeMap, structFieldsMap};
}

} // namespace kecc::ir
