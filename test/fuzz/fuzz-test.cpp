#include "FuzzConfig.h"
#include "FuzzConfig.h.in"
#include "kecc/driver/Compilation.h"
#include "kecc/driver/DriverConfig.h"
#include "kecc/utils/LogicalResult.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include <memory>
#include <regex>

#define SKIP_TEST 102

namespace kecc {

namespace cl {

struct FuzzOptions {
  llvm::cl::OptionCategory FuzzCategory{"Fuzzing Options"};

  llvm::cl::opt<int> numTests{
      "num",
      llvm::cl::desc(
          "Number of tests to run. 0 means infinite. Negative means error"),
      llvm::cl::init(0),
  };

  llvm::cl::alias numTestsA{
      "n",
      llvm::cl::desc("Alias for -num"),
      llvm::cl::aliasopt(numTests),
  };

  llvm::cl::opt<int> startSeed{
      "seed",
      "Provide seed of fuzz generation",
      llvm::cl::init(-1),
  };

  llvm::cl::alias startSeedA{
      "s",
      llvm::cl::desc("Alias for -seed"),
      llvm::cl::aliasopt(startSeed),
  };

  llvm::cl::opt<bool> reduce{
      "reduce",
      llvm::cl::desc("Reducing input file"),
      llvm::cl::init(false),
  };

  llvm::cl::alias reduceA{
      "r",
      llvm::cl::desc("Alias for -reduce"),
      llvm::cl::aliasopt(reduce),
  };

  llvm::cl::opt<bool> easy{
      "easy",
      llvm::cl::desc("Generate easier programs"),
      llvm::cl::init(false),
  };

  llvm::cl::opt<bool> clangAnalyze{
      "clang-analyze",
      llvm::cl::desc("Run clang static analyzer for reducing."),
      llvm::cl::init(false),
  };

  llvm::DenseMap<llvm::StringRef, llvm::StringRef> replaceDict{
      // clang-format off
      {"volatile ",                     ""},
      {"static ",                       ""},
      {"extern ",                       ""},
      {"__restrict",                    ""},
      {"long __undefined;",             ""},
      {"return 0;",                     "return (unsigned char)(crc32_context);"},
      {R"(__attribute__\s*\(\(.*\)\))", ""},
      {"_Float128",                     "double"},
      {"long double",                   "double"},
      {R"((\+[0-9^FL]*)L)",             R"(\1)"},
      {"union",                         "struct"},
      {R"(enum[\w\s]*\{[^\}]*\};)",     ""},
      {R"(typedef enum[\w\s]*\{[^;]*;[\s_A-Z]*;)",     ""},
      {R"(const char \*const sys_errlist\[\];)",       ""}, // remove ArraySize::Unknown 
      {R"([^\n]*printf[^;]*;)",         ""},
      {R"([^\n]*scanf[^;]*;)",          ""},
      {" restrict",                     ""},
      {"inline ",                       ""},
      {"_Nullable",                     ""},
      {R"("g_\w*", )",                  ""}, // remove StringLiteral used for print in transparent_crc
      {R"(char\* vname, )",             ""}, // remove parameters not used in transparent_crc
      {R"(transparent_crc_bytes\s*\([^;]*\);)",        ""}, // remove transparent_crc_bytes 
      {R"([^\n]*_IO_2_1_[^;]*;)",       ""}, // remove struct created by removing extern
      {R"(__asm\s*\([^\)]*\))",         ""}, // asm extension in mac
      {R"(__asm__\s*\([^\)]*\))",       ""}, // asm extension in linux
      {"typedef __builtin_va_list __gnuc_va_list;",    ""},
      {"typedef __gnuc_va_list va_list;",              ""},
      {R"(\(fabsf\()",                  "(("},
      // todo : need to consider the case below in the future:
      // avoid compile - time constant expressed as complex expression such as
      // `1 + 1`
      {"char _unused2[^;]*;",          "char _unused2[10];"},
      // clang-format on
  };
};

llvm::ManagedStatic<FuzzOptions> fuzzOptions;

void registerFuzzOptions() { *fuzzOptions; }

} // namespace cl

std::string generate(int seed, bool easy, llvm::StringRef outputPath) {
  llvm::SmallVector<llvm::StringRef> args{
      CSMITH_BIN_PATH,  "--no-argc",    "--no-arrays",        "--no-jumps",
      "--no-pointers",  "--no-structs", "--no-unions",        "--float",
      "--strict-float", "-seed",        std::to_string(seed),
  };

  if (easy) {
    args.append({"--max-block-depth", "2", "--max-block-size", "2",
                 "--max-struct-fields", "3"});
  }
  args.append({"-o", outputPath});

  auto csmithReturn = llvm::sys::ExecuteAndWait(CSMITH_BIN_PATH, args);
  if (csmithReturn != 0) {
    llvm::errs() << "Error: csmith failed with return code " << csmithReturn
                 << '\n';
    return "";
  }

  auto fileBuffer = llvm::MemoryBuffer::getFile(outputPath);
  if (fileBuffer.getError()) {
    llvm::errs() << "Error: could not read generated file\n";
    return "";
  }
  auto code = fileBuffer.get()->getBuffer().str();
  return code;
}

std::string polish(llvm::StringRef code, llvm::StringRef outputPath) {

  {
    std::error_code ec;
    llvm::raw_fd_ostream os(outputPath, ec);
    if (ec) {
      llvm::errs() << "Error: could not create temporary file\n";
      return "";
    }
    os << code;
  }

  llvm::SmallVector<llvm::StringRef> clangArgs{
      CLANG_DIR, outputPath, "-I", CSMITH_INCLUDE_PATH,
      "-E",      "-P",       "-o", outputPath,
  };

  auto clangReturn = llvm::sys::ExecuteAndWait(CLANG_DIR, clangArgs);
  if (clangReturn != 0) {
    llvm::errs() << "Error: clang failed with return code " << clangReturn
                 << '\n';
    return "";
  }

  auto fileBuffer = llvm::MemoryBuffer::getFile(outputPath);
  if (fileBuffer.getError()) {
    llvm::errs() << "Error: could not read preprocessed file\n";
    return "";
  }
  auto polishedCode = fileBuffer.get()->getBuffer().str();

  for (const auto &[pattern, replace] : cl::fuzzOptions->replaceDict) {
    polishedCode = std::regex_replace(polishedCode, std::regex(pattern.str()),
                                      replace.str());
  }

  {
    std::error_code ec;
    llvm::raw_fd_ostream os(outputPath, ec);
    if (ec) {
      llvm::errs() << "Error: could not create temporary file\n";
      return "";
    }
    os << polishedCode;
  }
  return polishedCode;
}

int fuzz(llvm::StringRef testDir, int numTests, bool easy) {
  size_t i = 0;
  size_t skipped = 0;

  if (numTests == 0) {
    llvm::outs() << "Fuzzing with infinitly many test cases.  Please press "
                    "[Ctrl+C] to break.\n";
  } else {
    llvm::outs() << "Fuzzing with " << numTests << " test cases.\n";
  }

  while (1) {
    llvm::outs() << std::format("Test case #{} (skipped: {})\n", i + 1,
                                skipped);

    llvm::SmallVector<char> testC;
    llvm::sys::path::append(testC, testDir, "test.c");

    auto seed = rand() % 987654321 + 1;
    auto code =
        generate(seed, easy, llvm::StringRef{testC.data(), testC.size()});

    if (code.empty()) {
      llvm::errs() << "Error: code generation failed\n";
      return 1;
    }

    llvm::SmallVector<char> testP;
    llvm::sys::path::append(testP, testDir, "test_polished.c");
    code = polish(code, llvm::StringRef{testP.data(), testP.size()});
    if (code.empty()) {
      llvm::errs() << "Error: code polishing failed\n";
      return 1;
    }

    llvm::SmallVector<llvm::StringRef> fuzzArgs{
        FUZZ_BIN_PATH,
        llvm::StringRef{testP.data(), testP.size()},
    };

    auto fuzzReturn = llvm::sys::ExecuteAndWait(FUZZ_BIN_PATH, fuzzArgs,
                                                std::nullopt, std::nullopt, 60);
    if (fuzzReturn == SKIP_TEST) {
      ++skipped;
      continue;
    } else if (fuzzReturn != 0) {
      llvm::errs() << std::format("Test `{}` failed with exit code {}.\n",
                                  llvm::join(fuzzArgs, " "), fuzzReturn);
      return 1;
    }

    ++i;
    if (numTests != 0 && i >= static_cast<size_t>(numTests))
      return 0;
  }
}

std::string makeFuzzErrmsg(llvm::StringRef testDir) {
  llvm::SmallVector<char> testP;
  llvm::sys::path::append(testP, testDir, "test_polished.c");

  TempDirectory tempDir;

  llvm::SmallVector<char> stdOut;
  llvm::sys::path::append(stdOut, tempDir.getDirectory(), "stdout.txt");
  llvm::StringRef stdOutRef{stdOut.data(), stdOut.size()};

  llvm::SmallVector<llvm::StringRef> fuzzArgs{
      FUZZ_BIN_PATH,
      llvm::StringRef{testP.data(), testP.size()},
  };

  auto fuzzReturn =
      llvm::sys::ExecuteAndWait(FUZZ_BIN_PATH, fuzzArgs, std::nullopt,
                                {std::nullopt, stdOutRef, stdOutRef}, 60);
  llvm::StringRef errMsg = "panicked";
  if (fuzzReturn != 0) {
    auto fileBuffer = llvm::MemoryBuffer::getFile(stdOutRef);
    if (fileBuffer.getError()) {
      llvm::errs() << "Error: could not read stdout file\n";
      return "";
    }
    auto output = fileBuffer.get()->getBuffer();
    if (output.find("assert") == llvm::StringRef::npos) {
      errMsg = "assertion failed";
    }
  }
  return errMsg.str();
}

utils::LogicalResult makeReduceCriteria(llvm::StringRef testDir,
                                        llvm::StringRef errMsg,
                                        bool clangAnalyze) {
  auto templateBuffer =
      llvm::MemoryBuffer::getFile(REDUCE_CRITERIA_TEMPLATE_PATH);
  if (templateBuffer.getError()) {
    llvm::errs() << "Error: could not read reduce criteria template file\n";
    return utils::LogicalResult::failure();
  }

  auto templateStr = templateBuffer.get()->getBuffer();
  llvm::SmallVector<char> criteriaPath;
  llvm::sys::path::append(criteriaPath, testDir, "reduce-criteria.sh");

  std::error_code ec;
  llvm::raw_fd_ostream os(
      llvm::StringRef(criteriaPath.data(), criteriaPath.size()), ec);
  if (ec) {
    llvm::errs() << "Error: could not create reduce criteria file\n";
    return utils::LogicalResult::failure();
  }

  size_t pos = 0;
  while (pos != llvm::StringRef::npos) {
    auto nextPos = templateStr.find('$', pos);
    if (nextPos == llvm::StringRef::npos) {
      os << templateStr.substr(pos);
      break;
    }

    os << templateStr.slice(pos, nextPos);
    templateStr = templateStr.drop_front(nextPos);
    if (templateStr.starts_with("$TEST_DIR")) {
      pos = nextPos + 9;
      os << testDir;
    } else if (templateStr.starts_with("$REDUCED_C")) {
      pos = nextPos + 10;
      llvm::SmallVector<char> reducedC;
      llvm::sys::path::append(reducedC, testDir, "test_reduced.c");
      os << llvm::StringRef{reducedC.data(), reducedC.size()};
    } else if (templateStr.starts_with("$FUZZ_BIN")) {
      pos = nextPos + 8;
      os << FUZZ_BIN_PATH;
    } else if (templateStr.starts_with("$FUZZ_ERRMSG")) {
      pos = nextPos + 12;
      os << errMsg;
    } else if (templateStr.starts_with("$CLANG_ANALYZE")) {
      pos = nextPos + 15;
      if (clangAnalyze)
        os << "true";
      else
        os << "false";
    } else {
      os << '$';
      pos = nextPos + 1;
    }
  }

  ec = llvm::sys::fs::setPermissions(
      llvm::StringRef{criteriaPath.data(), criteriaPath.size()},
      llvm::sys::fs::perms::owner_exe);
  if (ec) {
    llvm::errs() << "Error: could not set execute permission to reduce "
                    "criteria file\n";
    return utils::LogicalResult::failure();
  }

  return utils::LogicalResult::success();
}

int creduce(llvm::StringRef testDir, bool clangAnalyze) {
  auto errMsg = makeFuzzErrmsg(testDir);
  if (errMsg.empty()) {
    llvm::errs() << "Error: could not reproduce the error\n";
    return 1;
  }
  if (makeReduceCriteria(testDir, errMsg, clangAnalyze).failed())
    return 1;

  llvm::SmallVector<char> testP;
  llvm::sys::path::append(testP, testDir, "test_polished.c");
  llvm::SmallVector<char> testR;
  llvm::sys::path::append(testR, testDir, "test_reduced.c");
  std::error_code ec =
      llvm::sys::fs::copy_file(llvm::StringRef{testP.data(), testP.size()},
                               llvm::StringRef{testR.data(), testR.size()});
  if (ec) {
    llvm::errs() << "Error: could not copy polished file to reduced file\n";
    return 1;
  }

  llvm::SmallVector<char> criteria;
  llvm::sys::path::append(criteria, testDir, "reduce-criteria.sh");
  llvm::SmallVector<llvm::StringRef> creduceArgs{
      CREDUCE_BIN_PATH,
      "--tidy",
      "--timing",
      "--timeout",
      "20",
      llvm::StringRef{criteria.data(), criteria.size()},
      llvm::StringRef{testR.data(), testR.size()},
  };

  TempDirectory tempDir;
  llvm::SmallVector<char> creduceLog;
  llvm::sys::path::append(creduceLog, tempDir.getDirectory(), "creduce.log");
  llvm::StringRef creduceLogRef{creduceLog.data(), creduceLog.size()};

  auto creduceReturn = llvm::sys::ExecuteAndWait(
      CREDUCE_BIN_PATH, creduceArgs, std::nullopt,
      {std::nullopt, creduceLogRef, creduceLogRef}, 0);
  if (creduceReturn != 0) {
    auto fileBuffer = llvm::MemoryBuffer::getFile(creduceLogRef);
    if (fileBuffer.getError()) {
      llvm::errs() << "Error: could not read creduce log file\n";
      return 1;
    }
    auto output = fileBuffer.get()->getBuffer();
    llvm::outs() << output;
    llvm::errs() << "Error: creduce failed with return code " << creduceReturn
                 << '\n';
    return 1;
  }
  llvm::outs() << "Reduction completed. See "
               << llvm::StringRef{testR.data(), testR.size()}
               << " for the reduced test case.\n";
  return 0;
}

int fuzz_test_main(int argc, char **argv) {
  cl::registerFuzzOptions();

  llvm::cl::ParseCommandLineOptions(argc, argv, "kecc fuzz tester\n");

  if (cl::fuzzOptions->numTests < 0) {
    llvm::errs() << "Error: number of tests must be non-negative\n";
    return 1;
  }

  if (cl::fuzzOptions->startSeed != -1)
    srand(cl::fuzzOptions->startSeed);
  else
    llvm::outs() << "Use default random seed";

  llvm::SmallVector<char> fuzzDir;
  llvm::sys::path::append(fuzzDir, TEST_LOG_DIR, "fuzz");

  if (cl::fuzzOptions->reduce) {
    return creduce(llvm::StringRef{fuzzDir.data(), fuzzDir.size()},
                   cl::fuzzOptions->clangAnalyze);
  } else {
    return fuzz(llvm::StringRef{fuzzDir.data(), fuzzDir.size()},
                cl::fuzzOptions->numTests, cl::fuzzOptions->easy);
  }
}

} // namespace kecc

int main(int argc, char **argv) {
  llvm::InitLLVM X(argc, argv);
  return kecc::fuzz_test_main(argc, argv);
}
