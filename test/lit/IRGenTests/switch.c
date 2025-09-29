// RUN: kecc %s -S -emit-kecc -print-stdout | FileCheck %s
// RUN: keci %s --test-return-value=1
// clang-format off

// CHECK: fun i32 @main () {
// CHECK-NEXT: init:
// CHECK-NEXT:   bid: b0
// CHECK-NEXT:   allocations:
// CHECK-NEXT:    %l0:i32:a
// CHECK-NEXT:    %l1:i32:b
int main() {
  // CHECK-LABEL: block b0:
  int a = 1;
  // CHECK-NEXT:   %b0:i0:unit = store 1:i32 %l0:i32*
  int b = 0;
  // CHECK-NEXT:   %b0:i1:unit = store 0:i32 %l1:i32*

  // CHECK-NEXT:   %b0:i2:i32 = load %l0:i32*
  // CHECK-NEXT:   switch %b0:i2:i32 default b1() [
  // CHECK-NEXT:     0:i32 b3()
  // CHECK-NEXT:     1:i32 b4()
  // CHECK-NEXT:   ]

  // CHECK-LABEL: block b1:
  // CHECK-NEXT:   %b1:i0:i32 = load %l1:i32*
  // CHECK-NEXT:   %b1:i1:i32 = add %b1:i0:i32 3:i32
  // CHECK-NEXT:   %b1:i2:unit = store %b1:i1:i32 %l1:i32*
  // CHECK-NEXT:   j b2()

  // CHECK-LABEL: block b2:
  // CHECK-NEXT:   %b2:i0:i32 = load %l1:i32*
  // CHECK-NEXT:   %b2:i1:i1 = cmp eq %b2:i0:i32 2:i32
  // CHECK-NEXT:   %b2:i2:i32 = typecast %b2:i1:i1 to i32
  // CHECK-NEXT:   ret %b2:i2:i32

  // CHECK-LABEL: block b3:
  // CHECK-NEXT:   %b3:i0:i32 = load %l1:i32*
  // CHECK-NEXT:   %b3:i1:i32 = add %b3:i0:i32 1:i32
  // CHECK-NEXT:   %b3:i2:unit = store %b3:i1:i32 %l1:i32*
  // CHECK-NEXT:   j b2()

  // CHECK-LABEL: block b4:
  // CHECK-NEXT:   %b4:i0:i32 = load %l1:i32*
  // CHECK-NEXT:   %b4:i1:i32 = add %b4:i0:i32 2:i32
  // CHECK-NEXT:   %b4:i2:unit = store %b4:i1:i32 %l1:i32*
  // CHECK-NEXT:   j b2()
  switch (a) {
  case 0: {
    b += 1;
    break;
  }
  case 1: {
    b += 2;
    break;
  }
  default: {
    b += 3;
    break;
  }
  }

  return b == 2;
}
