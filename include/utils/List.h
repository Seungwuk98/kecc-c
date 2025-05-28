#ifndef KECC_UTILS_LIST_H
#define KECC_UTILS_LIST_H

#include <utility>
namespace kecc::utils {

template <typename T> class ListObject {
protected:
  struct Node {
    Node *next = nullptr;
    Node *prev = nullptr;
    T data = nullptr;

    ~Node() {
      if (next) {
        delete next;
      }
    }

    void insertNext(T newData) {
      Node *newNode = new Node();
      newNode->data = std::move(newData);
      next->prev = newNode;
      newNode->next = next;
      newNode->prev = this;
      next = newNode;
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

    T operator*() const { return curr ? curr->inst : nullptr; }
    bool operator!=(const Iterator &other) const { return curr != other.curr; }
    bool operator==(const Iterator &other) const { return curr == other.curr; }
    explicit operator bool() const { return curr != nullptr; }

    Node *getNode() const { return curr; }

  private:
    Node *curr;
  };

public:
  ListObject() {
    head = new Node();
    tail = new Node();
    head->next = tail;
    tail->prev = head;
  }
  ~ListObject() { delete head; }
  Node *getHead() const { return head; }
  Node *getTail() const { return tail; }

  Iterator begin() const { return Iterator(getHead()); }
  Iterator end() const { return Iterator(getTail()); }

private:
  Node *head = nullptr;
  Node *tail = nullptr;
};
} // namespace kecc::utils

#endif // KECC_UTILS_LIST_H
