#ifndef KECC_UTILS_LOGICALRESULT_H
#define KECC_UTILS_LOGICALRESULT_H

namespace kecc::utils {

class LogicalResult {
public:
  enum Status {
    Success,
    Failure,
    Error,
  };

  LogicalResult() {}
  LogicalResult(Status status) : status(status) {}

  static LogicalResult success() { return LogicalResult(Success); }
  static LogicalResult failure() { return LogicalResult(Failure); }
  static LogicalResult error() { return LogicalResult(Error); }

  bool succeeded() const { return status == Success; }
  bool failed() const { return status == Failure; }
  bool isError() const { return status == Error; }

private:
  Status status;
};

} // namespace kecc::utils

#endif // KECC_UTILS_LOGICALRESULT_H
