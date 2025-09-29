// RUN: kecc %s -S -emit-kecc -print-stdout | FileCheck %s
// RUN: keci %s --test-return-value=21
// clang-format off


int nonce = 1; // For random input
// CHECK: var i32 @nonce = 1
int g = 10;
// CHECK: var i32 @g = 10

// CHECK: fun i32 @foo (i32, i32) {
// CHECK-NEXT: init:
// CHECK-NEXT:   bid: b0
// CHECK-NEXT:   allocations:
// CHECK-NEXT:     %l0:i32:i 
// CHECK-NEXT:     %l1:i32:j 

// CHECK-LABEL: block b0:
// CHECK-NEXT:   %b0:p0:i32:i 
// CHECK-NEXT:   %b0:p1:i32:j 
// CHECK-NEXT:   %b0:i0:unit = store %b0:p0:i32 %l0:i32* 
// CHECK-NEXT:   %b0:i1:unit = store %b0:p1:i32 %l1:i32* 
// CHECK-NEXT:   %b0:i2:i32 = load %l0:i32* 
// CHECK-NEXT:   %b0:i3:i32 = load %l1:i32* 
// CHECK-NEXT:   %b0:i4:i32 = add %b0:i2:i32 %b0:i3:i32 
// CHECK-NEXT:   %b0:i5:i32 = load @nonce:i32* 
// CHECK-NEXT:   %b0:i6:i32 = add %b0:i4:i32 %b0:i5:i32
// CHECK-NEXT:   ret %b0:i6:i32
// CHECK-NEXT: }
int foo(int, int k);

// CHECK: fun i32 @main () {
// CHECK-NEXT: init:
// CHECK-NEXT:   bid: b0 
// CHECK-NEXT:   allocations: 
// CHECK-NEXT:     %l0:i32:i
int main() {
  // CHECK-LABEL: block b0:
  int i = g;
  // CHECK-NEXT:   %b0:i0:i32 = load @g:i32*
  // CHECK-NEXT:   %b0:i1:unit = store %b0:i0:i32 %l0:i32*

  // CHECK-NEXT:   %b0:i2:i32 = load %l0:i32*
  // CHECK-NEXT:   %b0:i3:i32 = load %l0:i32* 
  // CHECK-NEXT:   %b0:i4:i32 = call @foo:[ret:i32 params:(i32, i32)]*(%b0:i2:i32, %b0:i3:i32)
  // CHECK-NEXT:   ret %b0:i4:i32
  return foo(i, i);
}
// CHECK-NEXT: }

int foo(int i, int j) { return i + j + nonce; }
