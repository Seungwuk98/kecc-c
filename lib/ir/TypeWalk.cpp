#include "kecc/ir/TypeWalk.h"
#include "kecc/ir/Attribute.h"
#include "kecc/ir/Type.h"
#include "kecc/ir/WalkSupport.h"
#include "kecc/utils/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"

namespace kecc::ir {

class TypeWalkerDetail {
public:
  TypeWalkerDetail(const TypeWalker &walker) : walker(walker) {}

  template <typename T> WalkResult walkImpl(T obj);

private:
  template <typename T> WalkResult walkSubElements(T obj);

  WalkResult applyFns(Type type);
  WalkResult applyFns(Attribute type);

  const TypeWalker &walker;
};

WalkResult TypeWalkerDetail::applyFns(Type type) {
  for (auto &fn : walker.typeWalkFns) {
    auto result = fn(type);
    if (!result.isAdvance())
      return result;
  }

  return WalkResult::advance();
}

WalkResult TypeWalkerDetail::applyFns(Attribute attr) {
  for (auto &fn : walker.attrWalkFns) {
    auto result = fn(attr);
    if (!result.isAdvance())
      return result;
  }

  return WalkResult::advance();
}

template <typename T> WalkResult TypeWalkerDetail::walkImpl(T obj) {
  if (!obj)
    return WalkResult::interrupt();

  if (walker.order == TypeWalker::Order::PostOrder) {
    auto result = walkSubElements(obj);
    if (result.isInterrupt())
      return result;
  }

  WalkResult result = applyFns(obj);
  if (result.isInterrupt())
    return result;
  if (result.isSkip())
    return WalkResult::advance();

  if (walker.order == TypeWalker::Order::PreOrder) {
    auto walkResult = walkSubElements(obj);
    if (walkResult.isInterrupt())
      return walkResult;
  }

  return WalkResult::advance();
}

template <typename T> WalkResult TypeWalkerDetail::walkSubElements(T obj) {
  WalkResult result = WalkResult::advance();

  auto walkFn = [&](auto obj) {
    if (result.isInterrupt())
      return;

    auto walkResult = walkImpl(obj);
    if (walkResult.isInterrupt())
      result = walkResult;
  };

  obj.walkSubElements(walkFn, walkFn);
  return result;
}

WalkResult TypeWalker::walk(Type type) const {
  TypeWalkerDetail detail(*this);
  return detail.walkImpl(type);
}

WalkResult TypeWalker::walk(Attribute attr) const {
  TypeWalkerDetail detail(*this);
  return detail.walkImpl(attr);
}

class TypeReplacerDetail {
public:
  TypeReplacerDetail(TypeReplacer &replacer) : replacer(replacer) {}

  template <typename T> T replaceImpl(T obj);

private:
  std::optional<Type> applyFns(Type type);
  std::optional<Attribute> applyFns(Attribute attr);

  template <typename T>
  utils::LogicalResult
  replaceSubElements(T obj, llvm::SmallVectorImpl<Type> &replacedTypes,
                     llvm::SmallVectorImpl<Attribute> &replacedAttrs);

  TypeReplacer &replacer;
};

template <typename T> T TypeReplacerDetail::replaceImpl(T obj) {
  if (!obj)
    return nullptr;

  if (auto [it, inserted] =
          replacer.visited.try_emplace(obj.getImpl(), obj.getImpl());
      !inserted)
    return T::getFromVoidPointer(it->second);

  std::optional<T> replaced = applyFns(obj);

  if (!replaced)
    return nullptr;

  llvm::SmallVector<Type> replacedTypes;
  llvm::SmallVector<Attribute> replacedAttributes;
  if (replaceSubElements(*replaced, replacedTypes, replacedAttributes).failed())
    return nullptr;

  auto replacedType =
      replaced->replaceSubElements(replacedTypes, replacedAttributes);
  assert(replaced && "Replaced type must not be null");

  replacer.visited[obj.getImpl()] = replacedType.getImpl();
  return replacedType;
}

std::optional<Type> TypeReplacerDetail::applyFns(Type type) {
  for (auto &fn : replacer.replaceTypeFn) {
    auto [replaced, result] = fn(type);
    if (result.failed())
      continue;
    if (result.isError())
      return std::nullopt;

    assert(replaced && "Replaced type must not be null");
    if (replaced)
      return replaced;
  }

  return type;
}

std::optional<Attribute> TypeReplacerDetail::applyFns(Attribute attr) {
  for (auto &fn : replacer.replaceAttrFn) {
    auto [replaced, result] = fn(attr);
    if (result.failed())
      continue;
    if (result.isError())
      return std::nullopt;

    assert(replaced && "Replaced attribute must not be null");
    if (replaced)
      return replaced;
  }

  return attr;
}

template <typename T>
utils::LogicalResult TypeReplacerDetail::replaceSubElements(
    T obj, llvm::SmallVectorImpl<Type> &replacedTypes,
    llvm::SmallVectorImpl<Attribute> &replacedAttrs) {
  bool failed = false;
  obj.walkSubElements(
      [&](Type subType) {
        if (failed)
          return;

        auto replaced = replaceImpl(subType);
        if (!replaced) {
          failed = true;
          return;
        }

        replacedTypes.emplace_back(replaced);
      },
      [&](Attribute subAttr) {
        if (failed)
          return;

        auto replaced = replaceImpl(subAttr);
        if (!replaced) {
          failed = true;
          return;
        }

        replacedAttrs.emplace_back(replaced);
      });

  if (failed)
    return utils::LogicalResult::failure();
  return utils::LogicalResult::success();
}

TypeReplacer::TypeReplacer() = default;
TypeReplacer::~TypeReplacer() = default;

Type TypeReplacer::replace(Type type) {
  TypeReplacerDetail detail(*this);
  return detail.replaceImpl(type);
}

Attribute TypeReplacer::replace(Attribute attr) {
  TypeReplacerDetail detail(*this);
  return detail.replaceImpl(attr);
}

void TypeReplacer::addReplaceFnImpl(
    llvm::function_ref<ReplaceResult<Type>(Type)> replaceFn) {
  replaceTypeFn.emplace_back(std::move(replaceFn));
}

void TypeReplacer::addReplaceFnImpl(
    llvm::function_ref<ReplaceResult<Attribute>(Attribute)> replaceFn) {
  replaceAttrFn.emplace_back(std::move(replaceFn));
}
} // namespace kecc::ir
