#ifndef KECC_DRIVER_COMPILATION_H
#define KECC_DRIVER_COMPILATION_H

#include "kecc/ir/Context.h"
#include "kecc/ir/Pass.h"
#include "kecc/translate/TranslateContext.h"
#include "kecc/utils/LogicalResult.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SourceMgr.h"

namespace kecc {

enum OptLevel {
  O0,
  O1,
};

enum class InputFormat {
  C,
  KeccIR,
};

enum class OutputFormat {
  KeccIR,
  Assembly,
  Object,
  Executable,
};

class CompileOptTable {
public:
  CompileOptTable()
      : optLevel(O0), inputFormat(InputFormat::C),
        outputFormat(OutputFormat::Executable) {}

  OptLevel getOptLevel() const { return optLevel; }
  void setOptLevel(OptLevel level) { optLevel = level; }

  InputFormat getInputFormat() const { return inputFormat; }
  void setInputFormat(InputFormat format) { inputFormat = format; }

  OutputFormat getOutputFormat() const { return outputFormat; }
  void setOutputFormat(OutputFormat format) { outputFormat = format; }

private:
  friend class Compilation;
  OptLevel optLevel;
  InputFormat inputFormat;
  OutputFormat outputFormat;
};

struct TempDirectory {
  TempDirectory();
  ~TempDirectory();

  llvm::StringRef getDirectory() const { return {dir.data(), dir.size()}; }

private:
  llvm::SmallVector<char> dir;
};

class Compilation {
public:
  Compilation(const CompileOptTable &opt, llvm::StringRef inputFileName,
              llvm::StringRef outputFileName, llvm::StringRef inputSource,
              llvm::SourceMgr &sourceMgr);

  llvm::StringRef getInputFileName() const { return inputFileName; }
  llvm::StringRef getOutputFileName() const { return outputFileName; }
  llvm::StringRef getInputSource() const { return inputSource; }
  llvm::SourceMgr &getSourceMgr() const { return sourceMgr; }

  ir::IRContext *getIRContext() const { return irContext.get(); }
  TranslateContext *getTranslateContext() const {
    return translateContext.get();
  }
  TempDirectory *getTempDirectory() const { return tempDirectory.get(); }

  CompileOptTable &getCompileOptTable() { return opt; }

  void createIRContext() { irContext = std::make_unique<ir::IRContext>(); }
  void createTranslateContext() {
    translateContext = std::make_unique<TranslateContext>();
  }
  void createTempDirectory() {
    tempDirectory = std::make_unique<TempDirectory>();
  }

  utils::LogicalResult verify() const;
  int compile();

  void setOptPipeline(std::function<void(ir::PassManager &)> pipeline) {
    optPipeline = pipeline;
  }

private:
  llvm::StringRef inputFileName;
  llvm::StringRef outputFileName;

  llvm::StringRef inputSource;
  llvm::SourceMgr &sourceMgr;

  std::function<void(ir::PassManager &)> optPipeline;
  std::unique_ptr<ir::IRContext> irContext;
  std::unique_ptr<TranslateContext> translateContext;
  std::unique_ptr<TempDirectory> tempDirectory;
  CompileOptTable opt;
};

} // namespace kecc

#endif // KECC_DRIVER_COMPILATION_H
