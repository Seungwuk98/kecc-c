#include "ExecutorConfig.h"
#include "ExecutorConfig.h.in"
#include "kecc/utils/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdio>
#include <cstdlib>

namespace kecc {

namespace cl {
static llvm::cl::opt<std::string> input(llvm::cl::Positional,
                                        llvm::cl::init("-"),
                                        llvm::cl::desc("<input file>"));

static llvm::cl::opt<int>
    dumpSource("dump-source", llvm::cl::desc("Dump splitted source files"));

static llvm::cl::opt<std::string>
    arg("arg", llvm::cl::desc("Argument for the program"));
} // namespace cl

struct TempDirectory {
  TempDirectory() {
    llvm::SmallVector<char> tempDir;
    llvm::sys::path::system_temp_directory(true, tempDir);
    llvm::sys::path::append(tempDir, "kecc");
    auto ec = llvm::sys::fs::createUniqueDirectory(tempDir, dir);
    assert(!ec && "failed to create temporary directory");
    (void)ec;
  }
  ~TempDirectory() {
    auto ec = llvm::sys::fs::remove_directories(dir);
    assert(!ec && "failed to remove temporary directory");
  }

  llvm::StringRef getDirectory() const { return {dir.data(), dir.size()}; }

private:
  llvm::SmallVector<char> dir;
};

class SourceSpliter {
public:
  SourceSpliter(llvm::StringRef name, llvm::StringRef source)
      : prevSourceName(name), buffer(source) {}

  llvm::ArrayRef<std::pair<llvm::StringRef, llvm::StringRef>> getSources() {
    return sources;
  }

  utils::LogicalResult split();

private:
  static constexpr char CHAR_EOF = EOF;

  char advance() {
    if (cursor < buffer.size())
      return buffer[cursor++];
    return CHAR_EOF;
  }

  char peek() {
    if (cursor < buffer.size())
      return buffer[cursor];
    return CHAR_EOF;
  }

  char prev() {
    if (cursor > 0)
      return buffer[cursor - 1];
    return CHAR_EOF;
  }

  utils::LogicalResult lexBlockComment();

  void eatLine() {
    char c = peek();
    while (c != CHAR_EOF && c != '\n') {
      advance();
      c = peek();
    }
    if (c == '\n')
      advance();
  }

  llvm::StringRef buffer;
  size_t cursor = 0;
  size_t savedCursor = 0;
  llvm::StringRef prevSourceName;

  llvm::SmallVector<std::pair<llvm::StringRef, llvm::StringRef>, 4> sources;
};

utils::LogicalResult SourceSpliter::split() {
  // find '\n' "////*"

  char c = peek();
  while (c != CHAR_EOF) {
    char pr = prev();
    advance();
    char pk = peek();
    if (c == '/' && pk == '*') {
      auto result = lexBlockComment();
      if (result.failed())
        return result;
    } else if (pr == '\n' && c == '/' && pk == '/') {
      advance(); // consume second '/'
      pk = peek();
      if (pk == '/') {
        auto prevSource = buffer.slice(savedCursor, cursor - 2);
        sources.emplace_back(prevSourceName, prevSource);
        advance(); // consume third '/'
        pk = peek();
        while (pk == '/') {
          advance();
          pk = peek();
        }
        savedCursor = cursor;
        eatLine();
        prevSourceName = buffer.slice(savedCursor, cursor).trim();
        savedCursor = cursor;
      }
    }

    c = peek();
  }

  auto lastSource = buffer.slice(savedCursor, cursor);
  sources.emplace_back(prevSourceName, lastSource);
  return utils::LogicalResult::success();
}

utils::LogicalResult SourceSpliter::lexBlockComment() {
  // '/' is already consumed
  advance(); // consume '*'

  char c = peek();
  while (c != CHAR_EOF) {
    advance();
    char pk = peek();

    if (c == '/' && pk == '*') {
      lexBlockComment();
    } else if (c == '*' && pk == '/') {
      advance(); // consume '/'
      return utils::LogicalResult::success();
    }

    c = peek();
  }

  llvm::errs() << "Error: EOF reached before closing block comment";
  // unterminated block comment
  return utils::LogicalResult::failure();
}

int keccTestExecutorMain() {
  auto inputBufferOrErr = llvm::MemoryBuffer::getFileOrSTDIN(cl::input);
  if (inputBufferOrErr.getError()) {
    llvm::errs() << "read file error: " << inputBufferOrErr.getError().message()
                 << '\n';
    return 1;
  }

  auto source = inputBufferOrErr.get()->getBuffer();
  auto filename = llvm::sys::path::filename(cl::input);
  SourceSpliter spliter(filename, source);
  auto success = spliter.split();
  if (success.failed()) {
    return 1;
  }

  llvm::ArrayRef<std::pair<llvm::StringRef, llvm::StringRef>> sources =
      spliter.getSources();

  if (cl::dumpSource.getNumOccurrences() > 0) {
    if (cl::dumpSource >= (int)sources.size()) {
      llvm::errs() << "Error: dump-source index out of range\n";
      return 1;
    }

    llvm::outs() << sources[cl::dumpSource].second << '\n';
    return 0;
  }

  TempDirectory tempDir;

  // Create temporary files
  auto dir = tempDir.getDirectory();

  llvm::SmallVector<llvm::SmallVector<char>> tempFileNames;
  tempFileNames.resize(sources.size());

  for (size_t i = 0; i < sources.size(); ++i) {
    const auto &[name, source] = sources[i];
    auto &tempFileName = tempFileNames[i];
    tempFileName.append(dir.begin(), dir.end());
    llvm::sys::path::append(tempFileName, name);
  }

  for (size_t i = 0; i < sources.size(); ++i) {
    const auto &[name, source] = sources[i];
    auto &tempFileName = tempFileNames[i];

    std::error_code ec;
    llvm::raw_fd_ostream os(
        llvm::StringRef{tempFileName.data(), tempFileName.size()}, ec);
    if (ec) {
      llvm::errs() << "Error: failed to open temporary file: " << ec.message()
                   << '\n';
      return 1;
    }
    os << source;
  }

  // compile ir to assembly
  auto irPath = tempFileNames[0];
  auto asmPath = irPath;
  llvm::sys::path::replace_extension(asmPath, "s");
  auto returnCode = llvm::sys::ExecuteAndWait(
      KECC_TRANSLATE_DIR, {
                              KECC_TRANSLATE_DIR,
                              llvm::StringRef{irPath.data(), irPath.size()},
                              "-o",
                              llvm::StringRef{asmPath.data(), asmPath.size()},
                          });

  if (returnCode != 0) {
    llvm::errs() << "Error: kecc-translate failed with return code "
                 << returnCode << '\n';
    return returnCode;
  }

  // compile other source files and assembly
  llvm::SmallVector<llvm::StringRef> clangArgs = {
      CLANG_DIR,
      {asmPath.data(), asmPath.size()},
      "--target=riscv64-unknown-linux-gnu",
      "-fuse-ld=lld",
      "-w",
      "-I",
      INCLUDE_DIR};

  for (size_t i = 1; i < sources.size(); ++i) {
    auto &tempFileName = tempFileNames[i];
    clangArgs.emplace_back(tempFileName.data(), tempFileName.size());
  }

  clangArgs.emplace_back("-o");

  auto exePath = irPath;
  llvm::sys::path::replace_extension(exePath, ".out");
  clangArgs.emplace_back(exePath.data(), exePath.size());

  returnCode = llvm::sys::ExecuteAndWait(CLANG_DIR, clangArgs);
  if (returnCode != 0) {
    llvm::errs() << "Error: clang failed with return code " << returnCode
                 << '\n';
    return returnCode;
  }

  llvm::SmallVector<llvm::StringRef> exeArgs{
      QEMU_RISCV64_STATIC,
      llvm::StringRef{exePath.data(), exePath.size()},
  };
  auto args = llvm::split(cl::arg, ' ');
  exeArgs.append(args.begin(), args.end());

  // run executable with qemu
  returnCode = llvm::sys::ExecuteAndWait(QEMU_RISCV64_STATIC, exeArgs);

  if (returnCode != 0) {
    llvm::errs() << "Error: execution failed with return code " << returnCode
                 << '\n';
    return returnCode;
  }

  return 0;
}

} // namespace kecc

int main(int argc, const char **argv) {
  llvm::InitLLVM x(argc, argv);

  llvm::cl::ParseCommandLineOptions(argc, argv, "kecc test executor\n");
  return kecc::keccTestExecutorMain();
}
