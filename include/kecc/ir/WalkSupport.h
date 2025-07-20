#ifndef KECC_IR_WALK_SUPPORT
#define KECC_IR_WALK_SUPPORT

namespace kecc::ir {

struct WalkResult {
  enum Status { Advance, Skip, Interrupt };
  WalkResult(Status status) : status(status) {}

  static WalkResult advance() { return WalkResult(Status::Advance); }
  static WalkResult skip() { return WalkResult(Status::Skip); }
  static WalkResult interrupt() { return WalkResult(Status::Interrupt); }

  bool isAdvance() const { return status == Status::Advance; }
  bool isSkip() const { return status == Status::Skip; }
  bool isInterrupt() const { return status == Status::Interrupt; }

  Status status;
};

} // namespace kecc::ir

#endif // KECC_IR_WALK_SUPPORT
