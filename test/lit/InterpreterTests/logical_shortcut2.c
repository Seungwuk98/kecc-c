// RUN: keci %s --test-return-value=3
// RUN: kecc %s -S -emit-kecc -o - | keci -input-format=ir --test-return-value=3

int main() {
  int i = 0;
  int j = 1;

  int k = (++i) || (j++);
  return k + i + j; // expect 1 + 1 + 1 = 3
}
