// RUN: kecc %s -S -emit-kecc -print-stdout | FileCheck %s
// clang-format off

// CHECK: fun i32 @fibonacci (i32) {
// CHECK-NEXT: init:
// CHECK-NEXT:   bid: b0
// CHECK-NEXT:   allocations:
// CHECK-NEXT:    %l0:i32:n 
int fibonacci(int n) {
  // CHECK-LABEL: block b0:
  // CHECK-NEXT:   %b0:p0:i32:n
  // CHECK-NEXT:   %b0:i0:unit = store %b0:p0:i32 %l0:i32*
  // CHECK-NEXT:   j b1()
  while (n + n) {
    // CHECK-LABEL: block b1:
    // CHECK-NEXT:   %b1:i0:i32 = load %l0:i32*
    // CHECK-NEXT:   %b1:i1:i32 = load %l0:i32*
    // CHECK-NEXT:   %b1:i2:i32 = add %b1:i0:i32 %b1:i1:i32
    // CHECK-NEXT:   %b1:i3:i1 = cmp ne %b1:i2:i32 0:i32
    // CHECK-NEXT:   br %b1:i3:i1, b2(), b3()

    // CHECK-LABEL: block b2:
    // CHECK-NEXT:   %b2:i0:i32 = load %l0:i32*
    // CHECK-NEXT:   ret %b2:i0:i32
    return n;
  }
  // CHECK-LABEL: block b3:
  // CHECK-NEXT:   ret undef:i32
}
// CHECK-NEXT: }

// CHECK: fun i32 @main () {
// CHECK-NEXT: init:
// CHECK-NEXT:   bid: b0
// CHECK-NEXT:   allocations:
int main() { 
  // CHECK-LABEL: block b0:
  // CHECK-NEXT:   ret 1:i32
  return 1; 
}
// CHECK-NEXT: }
