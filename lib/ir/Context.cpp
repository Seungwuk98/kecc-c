#include "kecc/ir/Context.h"
#include "kecc/ir/Attribute.h"
#include "kecc/ir/IRAttributes.h"
#include "kecc/ir/IRInstructions.h"
#include "kecc/ir/IRTypes.h"
#include "kecc/ir/Type.h"
#include "kecc/utils/Diag.h"
#include "llvm/Support/raw_ostream.h"

namespace kecc::ir {

IRContext::IRContext()
    : typeStorage(std::make_unique<TypeStorage>(this)), diagEngine(srcMgr) {
  // Register built-in types
  registerBuiltinTypes(this);

  // Register built-in attributes
  registerBuiltinAttributes(this);

  // Register built-in instructions
  inst::registerBuiltinInstructions(this);
}

IRContext::~IRContext() {
  for (void *allocation : allocations) {
    free(allocation);
  }
}

std::string RegisterId::toString() const {
  std::string result;
  llvm::raw_string_ostream ss(result);

  ss << '%';
  switch (getKind()) {
  case Arg:
    ss << 'b' << getBlockId() << ':' << 'p';
    break;
  case Temp:
    ss << 'b' << getBlockId() << ':' << 'i';
    break;
  case Alloc:
    ss << 'l';
    break;
  case Unresolved:
    ss << 'U';
    break;
  }

  ss << getRegId();
  return result;
}

void *IRContext::allocate(size_t size) {
  void *ptr = llvm::safe_malloc(size);
  allocations.emplace_back(ptr);
  return ptr;
}

RegisterId RegisterId::arg(llvm::SMRange range, int blockId,
                           std::size_t regId) {
  return RegisterId(range, blockId, Kind::Arg, regId);
}

RegisterId RegisterId::temp(llvm::SMRange range, int blockId,
                            std::size_t regId) {
  return RegisterId(range, blockId, Kind::Temp, regId);
}

RegisterId RegisterId::alloc(llvm::SMRange range, std::size_t regId) {
  return RegisterId(range, RegisterId::AllocBlockId, Kind::Alloc, regId);
}

RegisterId RegisterId::unresolved(llvm::SMRange range, std::size_t regId) {
  return RegisterId(range, 0, Kind::Unresolved, regId);
}

} // namespace kecc::ir
