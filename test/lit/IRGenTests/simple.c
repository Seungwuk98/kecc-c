// RUN: kecc %s -S -emit-kecc -print-stdout | FileCheck %s
// clang-format off

int nonce = 1; // For random input
// CHECK: var i32 @nonce = 1

// CHECK: fun i32 @main () {
// CHECK-NEXT: init:
// CHECK-NEXT:   bid: b0
// CHECK-NEXT:   allocations:
// CHECK-NEXT:     %l0:i32:x
int main() {
  // CHECK-LABEL: block b0:
  int x = nonce;
  // CHECK-NEXT:   %b0:i0:i32 = load @nonce:i32*
  // CHECK-NEXT:   %b0:i1:unit = store %b0:i0:i32 %l0:i32*
  return x;
  // CHECK-NEXT:   %b0:i2:i32 = load %l0:i32*
  // CHECK-NEXT:   ret %b0:i2:i32
}
// CHECK-NEXT: }
