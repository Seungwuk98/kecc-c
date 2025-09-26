// RUN: kecc %s -S -emit-kecc -print-stdout | FileCheck %s
// clang-format off

// CHECK: fun i32 @main () {
// CHECK-NEXT: init:
// CHECK-NEXT:   bid: b0
// CHECK-NEXT:   allocations:
// CHECK-NEXT:     %l0:i32:x
// CHECK-NEXT:     %l1:i32:y
int main() {
  // CHECK-LABEL: block b0:
  int x = 3;
  // CHECK-NEXT:   %b0:i0:unit = store 3:i32 %l0:i32*
  int y = sizeof(++x);
  // CHECK-NEXT:   %b0:i1:unit = store 4:i32 %l1:i32*

  return x;
  // CHECK-NEXT:   %b0:i2:i32 = load %l0:i32*
  // CHECK-NEXT:   ret %b0:i2:i32
}
// CHECK-NEXT: }
