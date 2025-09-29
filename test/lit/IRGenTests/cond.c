// RUN: kecc %s -S -emit-kecc -print-stdout | FileCheck %s
// RUN: keci %s --test-return-value=1
// clang-format off

// CHECK: fun i32 @main () {
// CHECK-NEXT: init:
// CHECK-NEXT:   bid: b0
// CHECK-NEXT:   allocations:
// CHECK-NEXT:     %l0:i32:y
// CHECK-NEXT:     %l1:i32:x
// CHECK-NEXT:     %l2:i32
int main() {
  // CHECK-LABEL: block b0:
  int y = 1;
  int x = 0;
  // CHECK-NEXT:   %b0:i0:unit = store 1:i32 %l0:i32*
  // CHECK-NEXT:   %b0:i1:unit = store 0:i32 %l1:i32*

  // CHECK-NEXT:   %b0:i2:i32 = load %l1:i32*
  // CHECK-NEXT:   %b0:i3:i32 = load %l0:i32*
  // CHECK-NEXT:   %b0:i4:i1 = cmp eq %b0:i2:i32 %b0:i3:i32
  // CHECK-NEXT:   %b0:i5:i32 = typecast %b0:i4:i1 to i32
  // CHECK-NEXT:   %b0:i6:i1 = cmp ne %b0:i5:i32 0:i32
  // CHECK-NEXT:   br %b0:i6:i1, b1(), b2()

  // CHECK-LABEL: block b1:
  // CHECK-NEXT:   %b1:i0:unit = store 2:i32 %l2:i32*
  // CHECK-NEXT:   j b3()

  // CHECK-LABEL: block b2:
  // CHECK-NEXT:   %b2:i0:unit = store 5:i32 %l2:i32*
  // CHECK-NEXT:   j b3()

  // CHECK-LABEL: block b3:
  // CHECK-NEXT:   %b3:i0:i32 = load %l2:i32*
  // CHECK-NEXT:   %b3:i1:i1 = cmp eq %b3:i0:i32 5:i32
  // CHECK-NEXT:   %b3:i2:i32 = typecast %b3:i1:i1 to i32
  // CHECK-NEXT:   ret %b3:i2:i32
  return ((x == y) ? 2 : 5) == 5;
}
// CHECK-NEXT: }
