#ifndef KECC_UTILS_LIST_H
#define KECC_UTILS_LIST_H

#include <iterator>
#include <utility>
namespace kecc::utils {

template <typename Container, typename Object>
concept HasDeleteObjectFn = requires(Container, Object obj) {
  { Container::deleteObject(obj) } -> std::convertible_to<void>;
};

template <typename ConcreteContainer, typename T> class ListObject {
protected:
  struct Node {
    Node *next = nullptr;
    Node *prev = nullptr;
    T data;

    ~Node() {
      if constexpr (HasDeleteObjectFn<ConcreteContainer, T>) {
        ConcreteContainer::deleteObject(data);
      } else if constexpr (std::is_pointer_v<T>) {
        if (data) {
          delete data;
        }
      }
    }

    Node *insertNext(T newData) {
      Node *newNode = new Node();
      newNode->data = std::move(newData);
      next->prev = newNode;
      newNode->next = next;
      newNode->prev = this;
      next = newNode;
      return newNode;
    }

    void remove() {
      assert(prev && "Cannot remove a node without previous pointer");
      assert(next && "Cannot remove a node without next pointer");
      prev->next = next;
      next->prev = prev;
      delete this;
    }
  };

  class Iterator {
  public:
    Iterator(Node *node) : curr(node) {}

    Iterator &operator++() {
      assert(curr && "Cannot increment end iterator");
      curr = curr->next;
      return *this;
    }

    Iterator operator++(int) {
      Iterator temp = *this;
      ++(*this);
      return temp;
    }

    Iterator &operator--() {
      assert(curr && "Cannot decrement begin iterator");
      curr = curr->prev;
      return *this;
    }

    Iterator operator--(int) {
      Iterator temp = *this;
      --(*this);
      return temp;
    }

    T &operator*() const {
      assert(curr && "Dereferencing null iterator");
      return curr->data;
    }
    bool operator!=(const Iterator &other) const { return curr != other.curr; }
    bool operator==(const Iterator &other) const { return curr == other.curr; }
    explicit operator bool() const { return curr != nullptr; }

    Node *getNode() const { return curr; }

  private:
    Node *curr;
  };
  using ReverseIterator = std::reverse_iterator<Iterator>;

public:
  ListObject() {
    head = new Node();
    tail = new Node();
    head->next = tail;
    tail->prev = head;
  }
  ~ListObject() {
    Node *curr = head;
    while (curr) {
      Node *nextNode = curr->next;
      delete curr;
      curr = nextNode;
    }
  }
  Node *getHead() const { return head; }
  Node *getTail() const { return tail; }

  Iterator begin() const { return Iterator(getHead()->next); }
  Iterator end() const { return Iterator(getTail()); }
  ReverseIterator rbegin() const { return Iterator(getTail()->prev); }
  ReverseIterator rend() const { return Iterator(getHead()); }

  Node *push(T data) { return tail->prev->insertNext(data); }

  bool empty() const { return head->next == tail; }

private:
  Node *head = nullptr;
  Node *tail = nullptr;
};

template <typename ConcreteContainer, typename T, typename Pred>
bool all_of(const ListObject<ConcreteContainer, T> &list, Pred pred) {
  for (const auto &item : list) {
    if (!pred(item)) {
      return false;
    }
  }
  return true;
}

template <typename ConcreteContainer, typename T, typename Pred>
bool any_of(const ListObject<ConcreteContainer, T> &list, Pred pred) {
  for (const auto &item : list) {
    if (pred(item)) {
      return true;
    }
  }
  return false;
}

} // namespace kecc::utils

#endif // KECC_UTILS_LIST_H
