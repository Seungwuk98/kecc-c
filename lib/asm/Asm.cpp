#include "kecc/asm/Asm.h"
#include "kecc/asm/AsmInstruction.h"
#include <format>

namespace kecc::as {

void Asm::print(llvm::raw_ostream &os) const {
  for (const auto *section : functions) {
    section->print(os, 0);
    os << '\n';
  }

  for (const auto *section : variables) {
    section->print(os, 0);
    os << '\n';
  }
}

void Function::print(llvm::raw_ostream &os, size_t indent) const {
  for (const auto *block : blocks) {
    block->print(os, indent);
    os << '\n';
  }
}

void Variable::print(llvm::raw_ostream &os, size_t indent) const {
  os << label << ":\n";
  for (const auto *directive : directives) {
    printIndent(os, indent + 1);
    os << directive->toString() << '\n';
  }
}

void Block::print(llvm::raw_ostream &os, size_t indent) const {
  if (!label.empty())
    os << label << ":\n";

  for (const auto *inst : *this) {
    printIndent(os, indent + 1);
    inst->print(os);
    os << '\n';
  }
}

std::string AlignDirective::toString() const {
  return std::format(".align\t{}", alignment);
}

std::string GloblDirective::toString() const {
  return std::format(".globl\t{}", label);
}

std::string SectionDirective::toString() const {
  std::string sectionName;
  switch (sectionType) {
  case SectionType::Text:
    sectionName = ".text";
    break;
  case SectionType::Data:
    sectionName = ".data";
    break;
  case SectionType::Rodata:
    sectionName = ".rodata";
    break;
  case SectionType::Bss:
    sectionName = ".bss";
    break;
  }

  return std::format(".section\t{}", sectionName);
}

std::string TypeDirective::toString() const {
  return std::format(".type\t{}, {}", label,
                     kind == Kind::Function ? "@function" : "@object");
}

std::string ByteDirective::toString() const {
  return std::format(".byte\t{}", value);
}

std::string HalfDirective::toString() const {
  return std::format(".half\t{}", value);
}

std::string WordDirective::toString() const {
  return std::format(".word\t{}", value);
}

std::string QuadDirective::toString() const {
  return std::format(".quad\t{}", value);
}

std::string ZeroDirective::toString() const {
  return std::format(".zero\t{}", size);
}

} // namespace kecc::as
