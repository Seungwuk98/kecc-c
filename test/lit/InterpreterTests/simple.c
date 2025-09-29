// clang-format off
// RUN: keci %s --test-return-value=15
// RUN: kecc %s -S -emit-kecc -o - | keci -input-format=ir --test-return-value=15

int main() { return 15; }
