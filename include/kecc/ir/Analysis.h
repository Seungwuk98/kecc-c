#ifndef KECC_IR_ANALYSIS_H
#define KECC_IR_ANALYSIS_H

namespace kecc::ir {

class Module;

class Analysis {
public:
  virtual ~Analysis() = default;
  Analysis(Module *module) : module(module) {}

  Module *getModule() const { return module; }

private:
  Module *module;
};

} // namespace kecc::ir

#endif // KECC_IR_ANALYSIS_H
