#ifndef KECC_ASM_H
#define KECC_ASM_H

#include "kecc/utils/List.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

namespace kecc::as {

constexpr size_t IndentSize = 2;

inline void printIndent(llvm::raw_ostream &os, size_t indent) {
  os << std::string(indent * IndentSize, ' ');
}

template <typename Declaration> class Section;
class Block;
class Variable;
class Directive;
class Function;
class Instruction;

class Asm {
public:
  void print(llvm::raw_ostream &os) const;
  ~Asm();

private:
  llvm::SmallVector<Section<Function> *> functions;
  llvm::SmallVector<Section<Variable> *> variables;
};

template <typename Declaration> class Section {
public:
  Section(llvm::ArrayRef<Directive *> directives, Declaration *declaration)
      : directives(directives.begin(), directives.end()),
        declaration(declaration) {}
  ~Section();

  llvm::ArrayRef<Directive *> getDirectives() const { return directives; }
  Declaration *getDeclaration() const { return declaration; }

  void print(llvm::raw_ostream &os, size_t indent) const;

private:
  llvm::SmallVector<Directive *> directives;
  Declaration *declaration;
};

class Function {
public:
  Function(llvm::ArrayRef<Block *> blocks)
      : blocks(blocks.begin(), blocks.end()) {}
  ~Function();

  llvm::ArrayRef<Block *> getBlocks() const { return blocks; }
  void print(llvm::raw_ostream &os, size_t indent) const;

private:
  llvm::SmallVector<Block *> blocks;
};

class Variable {
public:
  Variable(llvm::StringRef label, llvm::ArrayRef<Directive *> directives)
      : label(label.str()), directives(directives.begin(), directives.end()) {}

  ~Variable();
  void print(llvm::raw_ostream &os, size_t indent) const;

private:
  std::string label;
  llvm::SmallVector<Directive *> directives;
};

class Block : public utils::ListObject<Block, Instruction *> {
public:
  Block(llvm::StringRef label);
  ~Block();
  using Node = utils::ListObject<Block, Instruction *>::Node;

  llvm::StringRef getLabel() const { return label; }
  void print(llvm::raw_ostream &os, size_t indent = 0) const;

  class InsertionPoint {
  public:
    InsertionPoint() : block(nullptr), it(nullptr) {};
    InsertionPoint(Block *block, Iterator it) : block(block), it(it) {}

    InsertionPoint insertNext(Instruction *inst) {
      assert(it.getNode()->next && "Cannot insert after the tail of list");
      Node *newNode = it.getNode()->insertNext(inst);
      return InsertionPoint(block, Iterator(newNode));
    }

    Block *getBlock() const { return block; }
    Iterator getIterator() const { return it; }

    bool isValid() const { return block != nullptr && it.getNode() != nullptr; }

    InsertionPoint &operator++() {
      it++;
      return *this;
    }
    InsertionPoint operator++(int) {
      InsertionPoint temp = *this;
      ++(*this);
      return temp;
    }
    InsertionPoint &operator--() {
      it--;
      return *this;
    }
    InsertionPoint operator--(int) {
      InsertionPoint temp = *this;
      --(*this);
      return temp;
    }

  private:
    Block *block;
    Iterator it;
  };

private:
  std::string label;
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
  virtual std::string toString() const = 0;

  Kind getKind() const { return kind; }

protected:
  Directive(Kind kind) : kind(kind) {}

private:
  Kind kind;
};

class AlignDirective : public Directive {
public:
  AlignDirective(std::size_t alignment)
      : Directive(Kind::Align), alignment(alignment) {}

  std::string toString() const override;
  std::size_t getAlignment() const { return alignment; }

private:
  std::size_t alignment;
};

class GloblDirective : public Directive {
public:
  GloblDirective(llvm::StringRef label)
      : Directive(Kind::Globl), label(label) {}

  std::string toString() const override;
  llvm::StringRef getLabel() const { return label; }

private:
  std::string label;
};

class SectionDirective : public Directive {
public:
  enum SectionType {
    Text,
    Data,
    Rodata,
    Bss,
  };

  SectionDirective(SectionType sectionType)
      : Directive(Kind::Section), sectionType(sectionType) {}

  std::string toString() const override;
  SectionType getSectionType() const { return sectionType; }

private:
  SectionType sectionType;
};

class TypeDirective : public Directive {
public:
  enum class Kind {
    Function,
    Object,
  };

  TypeDirective(llvm::StringRef label, Kind kind)
      : Directive(Directive::Kind::Type), label(label), kind(kind) {}

  std::string toString() const override;
  llvm::StringRef getLabel() const { return label; }
  Kind getKind() const { return kind; }

private:
  std::string label;
  Kind kind;
};

class ByteDirective : public Directive {
public:
  ByteDirective(std::int8_t value) : Directive(Kind::Byte), value(value) {}

  std::string toString() const override;
  std::uint8_t getValue() const { return value; }

private:
  std::uint8_t value;
};

class HalfDirective : public Directive {
public:
  HalfDirective(std::int16_t value) : Directive(Kind::Half), value(value) {}

  std::string toString() const override;
  std::uint16_t getValue() const { return value; }

private:
  std::uint16_t value;
};

class WordDirective : public Directive {
public:
  WordDirective(std::int32_t value) : Directive(Kind::Word), value(value) {}
  std::string toString() const override;
  std::uint32_t getValue() const { return value; }

private:
  std::uint32_t value;
};

class QuadDirective : public Directive {
public:
  QuadDirective(std::int64_t value) : Directive(Kind::Quad), value(value) {}

  std::string toString() const override;
  std::uint64_t getValue() const { return value; }

private:
  std::uint64_t value;
};

class ZeroDirective : public Directive {
public:
  ZeroDirective(std::size_t size) : Directive(Kind::Zero), size(size) {}

  std::string toString() const override;

  std::size_t getSize() const { return size; }

private:
  std::size_t size;
};

template <typename Declaration> Section<Declaration>::~Section() {
  for (auto directive : directives)
    delete directive;
  delete declaration;
}

template <typename Declaration>
void Section<Declaration>::print(llvm::raw_ostream &os, size_t indent) const {
  for (const auto *directive : directives) {
    printIndent(os, indent + 1);
    os << directive->toString() << '\n';
  }
  declaration->print(os, indent);
}

} // namespace kecc::as

#endif // KECC_ASM_H
