#ifndef KECC_DRIVER_ACTION_H
#define KECC_DRIVER_ACTION_H

#include "kecc/asm/AsmInstruction.h"
#include "kecc/ir/Context.h"
#include "kecc/utils/LogicalResult.h"
#include "llvm/ADT/StringRef.h"
#include <memory>

namespace kecc {

class Compilation;

class ActionData : public utils::PointerCastBase<ActionData> {
public:
  ActionData(utils::LogicalResult result);
  virtual ~ActionData() = default;

  TypeID getTypeID() const { return typeId; }
  utils::LogicalResult getLogicalResult() const { return result; }

  static llvm::StringRef getNameStr() {
    return "Anonymous Action Result or Argument";
  }
  virtual llvm::StringRef getName() const { return getNameStr(); }

  void setResult(utils::LogicalResult res) { result = res; }

protected:
  ActionData(utils::LogicalResult result, TypeID typeId)
      : result(result), typeId(typeId) {}

private:
  utils::LogicalResult result;
  TypeID typeId;
};

template <typename ConcreteType> struct ActionDataTemplate : public ActionData {
  using Base = ActionDataTemplate<ConcreteType>;

  ActionDataTemplate(utils::LogicalResult result)
      : ActionData(result, TypeID::get<ConcreteType>()) {}

  static bool classof(const ActionData *result) {
    return result->getTypeID() == TypeID::get<ConcreteType>();
  }
};

class Action {
public:
  Action(Compilation *compilation) : compilation(compilation) {}

  virtual ~Action() = default;
  virtual llvm::StringRef getActionName() const { return "Anonymous Action"; };

  virtual void preExecute(ActionData *arg) {}
  virtual void postExecute(ActionData *result) {}

  virtual std::unique_ptr<ActionData>
  execute(std::unique_ptr<ActionData> arg) = 0;

  virtual llvm::StringRef getDescription() const { return ""; }

  Compilation *getCompilation() const { return compilation; }

private:
  Compilation *compilation;
};

template <typename ArgType, typename ResultType, bool Enable = true>
class ActionTemplate;

template <typename ResultType>
class ActionTemplate<ActionData, ResultType> : public Action {
public:
  using Base = ActionTemplate<ActionData, ResultType>;
  using Action::Action;
};

template <typename ArgType, typename ResultType>
class ActionTemplate<ArgType, ResultType> : public Action {
public:
  using Base = ActionTemplate<ArgType, ResultType>;
  using Action::Action;

  void preExecute(ActionData *arg) override final {
    if (!arg->isa<ArgType>()) {
      llvm::errs() << "Action " << getActionName()
                   << " expected argument of type " << ArgType::getNameStr()
                   << ", but got " << arg->getName() << "\n";
      return;
    }
    preExecute(arg->cast<ArgType>());
  }

  virtual void preExecute(ArgType *arg) {}

  void postExecute(ActionData *result) override final {
    if (!result->isa<ResultType>()) {
      llvm::errs() << "Action " << getActionName()
                   << " expected result of type " << ResultType::getNameStr()
                   << ", but got " << result->getName() << "\n";
      return;
    }
    postExecute(result->cast<ResultType>());
  }

  virtual void postExecute(ResultType *result) {}

  std::unique_ptr<ActionData>
  execute(std::unique_ptr<ActionData> arg) override final {
    if (!arg->isa<ArgType>()) {
      llvm::errs() << "Action " << getActionName()
                   << " expected argument of type " << ArgType::getNameStr()
                   << ", but got " << arg->getName() << "\n";
      return nullptr;
    }
    std::unique_ptr<ArgType> castedArg(arg.release()->cast<ArgType>());
    return execute(std::move(castedArg));
  }

  virtual std::unique_ptr<ResultType> execute(std::unique_ptr<ArgType> arg) = 0;
};

class Invocation {
public:
  void addAction(std::unique_ptr<Action> action) {
    actions.emplace_back(std::move(action));
  }

  template <typename ActionType, typename... Args>
  void addAction(Args &&...args) {
    auto action = std::make_unique<ActionType>(std::forward<Args>(args)...);
    actions.emplace_back(std::move(action));
  }

  llvm::StringRef printAllSchedule() const;

  std::unique_ptr<ActionData> executeAll();

private:
  std::vector<std::unique_ptr<Action>> actions;
};

} // namespace kecc

#endif // KECC_DRIVER_ACTION_H
