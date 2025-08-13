#ifndef KECC_TRANSLATE_REGISTER_ALLOCATION_H
#define KECC_TRANSLATE_REGISTER_ALLOCATION_H

#include "kecc/asm/Register.h"
#include "kecc/ir/IRAnalyses.h"
#include "kecc/translate/LiveRange.h"

namespace kecc {

class RegisterAllocation {
public:
  as::Register getRegister(ir::Value value) const;
  as::Register getRegister(LiveRange liveRange) const;

private:
};

} // namespace kecc

#endif // KECC_TRANSLATE_REGISTER_ALLOCATION_H
