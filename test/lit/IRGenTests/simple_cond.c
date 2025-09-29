// RUN: kecc %s -S -emit-kecc -print-stdout | FileCheck %s
// RUN: keci %s --test-return-value=1
// clang-format off

// CHECK: fun i32 @f (i32) {
// CHECK-NEXT: init:
// CHECK-NEXT:   bid: b0
// CHECK-NEXT:   allocations:
// CHECK-NEXT:     %l0:i32:x
int f(int x) {
  // CHECK-LABEL: block b0:
  // CHECK-NEXT:   %b0:p0:i32
  // CHECK-NEXT:   %b0:i0:unit = store %b0:p0:i32 %l0:i32*
  // CHECK-NEXT:   %b0:i1:i32 = load %l0:i32*
  // CHECK-NEXT:   %b0:i2:i32 = add %b0:i1:i32 8:i32
  // CHECK-NEXT:   ret %b0:i2:i32
  return x + 8; 
}
// CHECK-NEXT: }

// CHECK: fun i32 @main () {
// CHECK-NEXT: init:
// CHECK-NEXT:   bid: b0
// CHECK-NEXT:   allocations:
// CHECK-NEXT:     %l0:i32:x
// CHECK-NEXT:     %l1:i32:y
// CHECK-NEXT:     %l2:i32
// CHECK-NEXT:     %l3:i32
int main() {
  // CHECK-LABEL: block b0:
  int x = 0;
  // CHECK-NEXT:   %b0:i0:unit = store 0:i32 %l0:i32*
  int y = (x++ == 1) ? 1 : 2;
  // CHECK-NEXT:   %b0:i1:i32 = load %l0:i32*
  // CHECK-NEXT:   %b0:i2:i32 = add %b0:i1:i32 1:i32
  // CHECK-NEXT:   %b0:i3:unit = store %b0:i2:i32 %l0:i32*
  // CHECK-NEXT:   %b0:i4:i1 = cmp eq %b0:i1:i32 1:i32
  // CHECK-NEXT:   %b0:i5:i32 = typecast %b0:i4:i1 to i32
  // CHECK-NEXT:   %b0:i6:i1 = cmp ne %b0:i5:i32 0:i32
  // CHECK-NEXT:   br %b0:i6:i1, b1(), b2()
  
  // CHECK-LABEL: block b1:
  // CHECK-NEXT:   %b1:i0:unit = store 1:i32 %l2:i32*
  // CHECK-NEXT:   j b3()

  // CHECK-LABEL: block b2:
  // CHECK-NEXT:   %b2:i0:unit = store 2:i32 %l2:i32*
  // CHECK-NEXT:   j b3()
  
  // CHECK-LABEL: block b3:
  // CHECK-NEXT:   %b3:i0:i32 = load %l2:i32*
  // CHECK-NEXT:   %b3:i1:unit = store %b3:i0:i32 %l1:i32*
  // CHECK-NEXT:   %b3:i2:i32 = load %l0:i32*
  // CHECK-NEXT:   %b3:i3:i32 = load %l1:i32*
  // CHECK-NEXT:   %b3:i4:i1 = cmp lt %b3:i2:i32 %b3:i3:i32
  // CHECK-NEXT:   %b3:i5:i32 = typecast %b3:i4:i1 to i32
  // CHECK-NEXT:   %b3:i6:i1 = cmp ne %b3:i5:i32 0:i32
  // CHECK-NEXT:   br %b3:i6:i1, b4(), b5()

  // CHECK-LABEL: block b4:
  // CHECK-NEXT:   %b4:i0:i32 = load %l0:i32*
  // CHECK-NEXT:   %b4:i1:unit = store %b4:i0:i32 %l3:i32*
  // CHECK-NEXT:   j b6()

  // CHECK-LABEL: block b5:
  // CHECK-NEXT:   %b5:i0:unit = store 2:i32 %l3:i32*
  // CHECK-NEXT:   j b6()

  // CHECK-LABEL: block b6:
  // CHECK-NEXT:   %b6:i0:i32 = load %l3:i32*
  // CHECK-NEXT:   %b6:i1:i32 = call @f:[ret:i32 params:(i32)]*(%b6:i0:i32)
  // CHECK-NEXT:   %b6:i2:i1 = cmp eq %b6:i1:i32 9:i32
  // CHECK-NEXT:   %b6:i3:i32 = typecast %b6:i2:i1 to i32
  // CHECK-NEXT:   ret %b6:i3:i32
  return f((x < y) ? x : 2) == 9;
}
// CHECK-NEXT: }

