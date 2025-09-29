// RUN: kecc %s -S -emit-kecc -print-stdout | FileCheck %s
// RUN: keci %s --test-return-value=1
// clang-format off
typedef int i32_t;
typedef i32_t *p_i32_t;

// CHECK: fun i32 @main () {
// CHECK-NEXT: init:
// CHECK-NEXT:   bid: b0
// CHECK-NEXT:   allocations:
// CHECK-NEXT:    %l0:i32:a
// CHECK-NEXT:    %l1:i32*const:b
int main() {
  // CHECK-LABEL: block b0:
  i32_t a = 0;
  // CHECK-NEXT:   %b0:i0:unit = store 0:i32 %l0:i32*
  p_i32_t const b = &a;
  // CHECK-NEXT:   %b0:i1:unit = store %l0:i32* %l1:i32*const*
  *b = 1;
  // CHECK-NEXT:   %b0:i2:i32* = load %l1:i32*const*
  // CHECK-NEXT:   %b0:i3:unit = store 1:i32 %b0:i2:i32*

  // CHECK-NEXT:   %b0:i4:i32* = load %l1:i32*const*
  // CHECK-NEXT:   %b0:i5:i32 = load %b0:i4:i32*
  // CHECK-NEXT:   ret %b0:i5:i32
  return *b;
}
// CHECK-NEXT: }
