#ifndef KECC_ASM_H
#define KECC_ASM_H

#include "kecc/utils/List.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace kecc::as {

template <typename Declaration> class Section;
class Block;
class Variable;
class Directive;
class Function;
class Instruction;

class Asm {
public:
private:
  llvm::SmallVector<Section<Function> *> functions;
  llvm::SmallVector<Section<Variable> *> variables;
};

template <typename Declaration> class Section {
public:
private:
  llvm::SmallVector<Directive *> directives;
  Declaration *declaration;
};

class Function {
public:
private:
  llvm::SmallVector<Block *> blocks;
};

class Variable {
public:
private:
  llvm::StringRef label;
  llvm::SmallVector<Directive *> directives;
};

class Block : public utils::ListObject<Block, Instruction *> {
public:
  using Node = utils::ListObject<Block, Instruction *>::Node;

private:
  llvm::StringRef label;
};

class Directive {
public:
  enum class Kind {
    Align,
    Globl,
    Section,
    Type,
    Byte,
    Half,
    Word,
    Quad,
    Zero,
  };
  virtual ~Directive() = default;

private:
  Kind kind;
};

class AlignDirective : public Directive {
public:
private:
  std::size_t alignment;
};

class GloblDirective : public Directive {
public:
private:
  llvm::StringRef label;
};

class SectionDirective : public Directive {
public:
  enum SectionType {
    Text,
    Data,
    Rodata,
    Bass,
  };

private:
  SectionType sectionType;
};

class TypeDirective : public Directive {
public:
  enum class Kind {
    Function,
    Object,
  };

private:
  llvm::StringRef label;
  Kind kind;
};

} // namespace kecc::as

#endif // KECC_ASM_H
