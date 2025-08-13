#ifndef KECC_IR_PASS_H
#define KECC_IR_PASS_H

#include "kecc/ir/Module.h"

namespace kecc::ir {

struct PassResult {
  enum class Status {
    Success,
    Failure,
    Skip,
  };

  PassResult(Status status) : status(status) {}
  PassResult() : status(Status::Success) {}

  bool isSuccess() const { return status == Status::Success; }
  bool isFailure() const { return status == Status::Failure; }
  bool isSkip() const { return status == Status::Skip; }

  static PassResult success() { return PassResult(Status::Success); }
  static PassResult failure() { return PassResult(Status::Failure); }
  static PassResult skip() { return PassResult(Status::Skip); }

  Status getStatus() const { return status; }

  bool operator==(const PassResult &other) const {
    return status == other.status;
  }

  bool operator!=(const PassResult &other) const { return !(*this == other); }

private:
  Status status;
};

class Pass {
public:
  virtual ~Pass() = default;

  virtual PassResult run(Module *module) = 0;
  virtual void init(Module *module) {}
  virtual void exit(Module *module) {}
  virtual llvm::StringRef getPassArgument() const { return "ananymous_pass"; }
  virtual llvm::StringRef getDescription() const { return ""; }
  virtual void setOption(llvm::StringRef) {}
};

template <typename PassType> class IteratePass : public Pass {
public:
  template <typename... Args>
  IteratePass(std::size_t iterCount, Args &&...args) : iterCount(iterCount) {
    passInstance = std::make_unique<PassType>(std::forward<Args>(args)...);
  }

  PassResult run(Module *module) override {
    auto iter = 0;
    while (true) {
      if (iterCount && iter++ >= iterCount)
        break;

      auto result = passInstance->run(module);
      if (result.isSkip())
        break;
      if (result.isFailure())
        return result;
    }
    return PassResult::success();
  }

private:
  std::size_t iterCount;
  std::unique_ptr<Pass> passInstance;
};

class FuncPass : public Pass {
public:
  FuncPass() = default;

  virtual void init(Module *module, Function *fun) {};
  virtual PassResult run(Module *module, Function *fun) = 0;
  virtual void exit(Module *module, Function *fun) {};

  PassResult run(Module *module) override final {
    PassResult returnResult = PassResult::success();
    for (auto fun : *module->getIR()) {
      init(module, fun);
      auto result = run(module, fun);
      if (result.isFailure())
        return result;
      if (result.isSkip())
        returnResult = PassResult::skip();
      exit(module, fun);
    }
    return returnResult;
  }
};

class PassManager {
public:
  PassManager() = default;

  void addPass(Pass *pass) { passes.emplace_back(pass, ""); }

  void addPass(Pass *pass, llvm::StringRef options) {
    passes.emplace_back(pass, options);
  }

  template <typename PassType> void addPass(llvm::StringRef options = "");

  PassResult run(Module *module);

private:
  llvm::SmallVector<std::pair<Pass *, llvm::StringRef>> passes;
};

class PassPipelineParserImpl;
class PassPipelineParser {
public:
  PassPipelineParser(llvm::StringRef arg, llvm::StringRef desc);
  ~PassPipelineParser();

  void addToPassManager(PassManager &pm) const;

private:
  std::unique_ptr<PassPipelineParserImpl> impl;
};

Pass *registerPass(const std::function<std::unique_ptr<Pass>()> &passFn);

template <typename Pass, typename... Args> void registerPass(Args &&...args) {
  registerPass([&]() -> std::unique_ptr<Pass> {
    return std::make_unique<Pass>(std::forward<Args>(args)...);
  });
}

Pass *getPassByName(llvm::StringRef name);

template <typename PassType>
void PassManager::addPass(llvm::StringRef options) {
  auto passName = PassType::getPassName();
  Pass *pass = getPassByName(passName);
  assert(pass && "Pass is not registered");
  addPass(pass, options);
}

} // namespace kecc::ir

#endif // KECC_IR_PASS_H
