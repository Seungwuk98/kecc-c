#include "kecc/driver/Action.h"

namespace kecc {

ActionData::ActionData(utils::LogicalResult result)
    : result(result), typeId(TypeID::get<ActionData>()) {}

std::unique_ptr<ActionData> Invocation::executeAll() {
  std::unique_ptr<ActionData> currentArg = nullptr;

  for (const std::unique_ptr<Action> &action : actions) {
    action->preExecute(currentArg.get());
    currentArg = action->execute(std::move(currentArg));
    if (!currentArg->getLogicalResult().succeeded())
      return currentArg;
    action->postExecute(currentArg.get());
  }

  return std::move(currentArg);
}

} // namespace kecc
