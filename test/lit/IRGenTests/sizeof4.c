// RUN: kecc %s -S -emit-kecc -print-stdout | FileCheck %s
// clang-format off

// CHECK: fun i32 @main () {
// CHECK-NEXT: init:
// CHECK-NEXT:   bid: b0
// CHECK-NEXT:   allocations:
// CHECK-NEXT:     %l0:i32:a
// CHECK-NEXT:     %l1:i32:b
int main() {
  // CHECK-LABEL: block b0:
  int a = 3;
  // CHECK-NEXT:   %b0:i0:unit = store 3:i32 %l0:i32*
  int b = sizeof(!(a++));
  // CHECK-NEXT:   %b0:i1:unit = store 4:i32 %l1:i32*
  return a + b;
  // CHECK-NEXT:   %b0:i2:i32 = load %l0:i32*
  // CHECK-NEXT:   %b0:i3:i32 = load %l1:i32*
  // CHECK-NEXT:   %b0:i4:i32 = add %b0:i2:i32 %b0:i3:i32
  // CHECK-NEXT:   ret %b0:i4:i32
}
// CHECK-NEXT: }
