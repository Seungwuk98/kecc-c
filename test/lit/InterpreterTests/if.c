// RUN: keci %s --test-return-value=1
// RUN: kecc %s -S -emit-kecc -o - | keci -input-format=ir --test-return-value=1

int max(int a, int b) {
  if (a > b) {
    return a;
  } else {
    return b;
  }
}

int main() { return max(10, 20) == 20 && max(20, 10) == 20; }
