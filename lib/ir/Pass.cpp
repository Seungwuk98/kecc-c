#include "kecc/ir/Pass.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"

namespace kecc::ir {

static llvm::ManagedStatic<llvm::StringMap<std::unique_ptr<Pass>>> passRegistry;

Pass *registerPass(const std::function<std::unique_ptr<Pass>()> &passFn) {
  auto pass = passFn();
  assert(pass && "Pass factory function must return a valid Pass instance");
  auto name = pass->getPassName();
  auto [it, inserted] = passRegistry->try_emplace(name, std::move(pass));
  assert(inserted && "Pass with the same name already registered");
  (void)inserted; // Suppress unused variable warning
  return it->second.get();
}

Pass *getPassByName(llvm::StringRef name) {
  auto it = passRegistry->find(name);
  if (it != passRegistry->end()) {
    return it->second.get();
  }
  return nullptr;
}

namespace {

struct PassArg {
  PassArg() = default;
  PassArg(Pass *pass) : pass(pass) {}

  llvm::StringRef options;
  Pass *pass = nullptr;
};

} // namespace

} // namespace kecc::ir

namespace llvm::cl {

template <>
struct OptionValue<kecc::ir::PassArg>
    : OptionValueBase<kecc::ir::PassArg, true> {

  OptionValue() = default;
  OptionValue(const kecc::ir::PassArg &value) : value(value) {}

  void setValue(const kecc::ir::PassArg &value) { this->value = value; }
  const kecc::ir::PassArg &getValue() const { return value; }
  bool hasValue() const { return true; }

  kecc::ir::PassArg value;
};
} // namespace llvm::cl

namespace kecc::ir {

PassResult PassManager::run(Module *module) {
  PassResult returnResult = PassResult::success();
  for (auto [pass, options] : passes) {
    pass->setOption(options);
    pass->init(module);
    auto result = pass->run(module);
    if (result.isFailure())
      return result; // Stop on failure
    if (result.isSkip())
      returnResult = PassResult::skip();
    pass->exit(module);
  }
  return returnResult; // All passes succeeded
}

namespace {
class PassNameParser : public llvm::cl::parser<PassArg> {
public:
  PassNameParser(llvm::cl::Option &O) : llvm::cl::parser<PassArg>(O) {}

  void initialize();
  bool parse(llvm::cl::Option &O, llvm::StringRef ArgName,
             llvm::StringRef ArgValue, PassArg &Val);
  void printOptionInfo(const llvm::cl::Option &O, size_t GlobalWidth) const;
};

void PassNameParser::initialize() {
  llvm::cl::parser<PassArg>::initialize();

  for (const auto &entry : *passRegistry) {
    addLiteralOption(entry.getKey(), PassArg(entry.second.get()),
                     entry.getValue()->getDescription());
  }
}

bool PassNameParser::parse(llvm::cl::Option &o, llvm::StringRef argName,
                           llvm::StringRef argValue, PassArg &val) {
  if (llvm::cl::parser<PassArg>::parse(o, argName, argValue, val))
    return true; // fail
  val.options = argValue;
  return false; // success
}

void PassNameParser::printOptionInfo(const llvm::cl::Option &o,
                                     size_t globalWidth) const {
  if (o.hasArgStr()) {
    llvm::outs() << "  --" << o.ArgStr;
    o.printHelpStr(o.HelpStr, globalWidth, o.ArgStr.size() + 4);
  } else {
    llvm::outs() << o.HelpStr;
  }

  llvm::SmallVector<Pass *, 8> entries;
  entries.reserve(passRegistry->size());
  for (const auto &entry : *passRegistry)
    entries.push_back(entry.second.get());
  llvm::sort(entries, [](const Pass *a, const Pass *b) {
    return a->getPassName() < b->getPassName();
  });
  llvm::outs().indent(4) << "Passes:\n";
  for (const auto *pass : entries) {
    llvm::outs().indent(8) << pass->getPassName() << ": "
                           << pass->getDescription() << "\n";
  }
}

} // namespace

class PassPipelineParserImpl {
public:
  PassPipelineParserImpl(llvm::StringRef arg, llvm::StringRef desc)
      : passList(arg, llvm::cl::desc(desc)) {
    passList.setValueExpectedFlag(
        llvm::cl::ValueExpected::ValueOptional); // Allow empty pass list
  }

  llvm::cl::list<PassArg, bool, PassNameParser> passList;
};

PassPipelineParser::PassPipelineParser(llvm::StringRef arg,
                                       llvm::StringRef desc)
    : impl(std::make_unique<PassPipelineParserImpl>(arg, desc)) {}

PassPipelineParser::~PassPipelineParser() = default;

void PassPipelineParser::addToPassManager(PassManager &pm) const {
  for (const auto &passArg : impl->passList) {
    pm.addPass(passArg.pass, passArg.options);
  }
}

} // namespace kecc::ir
