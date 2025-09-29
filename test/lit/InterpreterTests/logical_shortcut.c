// RUN: keci %s --test-return-value=3
// RUN: kecc %s -S -emit-kecc -o - | keci -input-format=ir --test-return-value=3

int main() {
  int i = 0;
  int j = 1;

  int k = (++i) && (++j);
  int l = (--i) && (++j);
  return k + i + j + l; // expect 0 + 1 + 2 + 0 = 3
}
