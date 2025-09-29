// RUN: kecc %s -S -emit-kecc -print-stdout | FileCheck %s
// RUN: keci %s --test-return-value=1
// clang-format off

// CHECK: fun i32 @fibonacci (i32) {
// CHECK-NEXT: init:
// CHECK-NEXT:   bid: b0
// CHECK-NEXT:   allocations:
// CHECK-NEXT:     %l0:i32:n 
// CHECK-NEXT:     %l1:i32:i 
// CHECK-NEXT:     %l2:i32:t1 
// CHECK-NEXT:     %l3:i32:t2 
// CHECK-NEXT:     %l4:i32:next_term 
int fibonacci(int n) {
  // CHECK-LABEL: block b0:
  // CHECK-NEXT:   %b0:p0:i32:n 
  // CHECK-NEXT:   %b0:i0:unit = store %b0:p0:i32 %l0:i32*
  int i = 0;
  // CHECK-NEXT:   %b0:i1:unit = store 0:i32 %l1:i32*
  int t1 = 0, t2 = 1, next_term = 0;
  // CHECK-NEXT:   %b0:i2:unit = store 0:i32 %l2:i32*
  // CHECK-NEXT:   %b0:i3:unit = store 1:i32 %l3:i32*
  // CHECK-NEXT:   %b0:i4:unit = store 0:i32 %l4:i32*

  // CHECK-NEXT:   %b0:i5:i32 = load %l0:i32*
  // CHECK-NEXT:   %b0:i6:i1 = cmp lt %b0:i5:i32 2:i32
  // CHECK-NEXT:   %b0:i7:i32 = typecast %b0:i6:i1 to i32
  // CHECK-NEXT:   %b0:i8:i1 = cmp ne %b0:i7:i32 0:i32
  // CHECK-NEXT:   br %b0:i8:i1, b1(), b2()
  if (n < 2) {
    // CHECK-LABEL: block b1:
    // CHECK-NEXT:   %b1:i0:i32 = load %l0:i32*
    // CHECK-NEXT:   ret %b1:i0:i32
    return n;
  }
  // CHECK-LABEL: block b2:
  // CHECK-NEXT:   j b3()

  // CHECK-LABEL: block b3:
  i = 1;
  // CHECK-NEXT:   %b3:i0:unit = store 1:i32 %l1:i32*
  // CHECK-NEXT:   j b4()
  while (i < n) {
    // CHECK-LABEL: block b4:
    // CHECK-NEXT:   %b4:i0:i32 = load %l1:i32*
    // CHECK-NEXT:   %b4:i1:i32 = load %l0:i32*
    // CHECK-NEXT:   %b4:i2:i1 = cmp lt %b4:i0:i32 %b4:i1:i32
    // CHECK-NEXT:   %b4:i3:i32 = typecast %b4:i2:i1 to i32
    // CHECK-NEXT:   %b4:i4:i1 = cmp ne %b4:i3:i32 0:i32
    // CHECK-NEXT:   br %b4:i4:i1, b5(), b6()

    // CHECK-LABEL: block b5:
    next_term = t1 + t2;
    // CHECK-NEXT:   %b5:i0:i32 = load %l2:i32*
    // CHECK-NEXT:   %b5:i1:i32 = load %l3:i32*
    // CHECK-NEXT:   %b5:i2:i32 = add %b5:i0:i32 %b5:i1:i32
    // CHECK-NEXT:   %b5:i3:unit = store %b5:i2:i32 %l4:i32*
    
    t1 = t2;
    // CHECK-NEXT:   %b5:i4:i32 = load %l3:i32*
    // CHECK-NEXT:   %b5:i5:unit = store %b5:i4:i32 %l2:i32*

    t2 = next_term;
    // CHECK-NEXT:   %b5:i6:i32 = load %l4:i32*
    // CHECK-NEXT:   %b5:i7:unit = store %b5:i6:i32 %l3:i32*

    ++i;
    // CHECK-NEXT:   %b5:i8:i32 = load %l1:i32*
    // CHECK-NEXT:   %b5:i9:i32 = add %b5:i8:i32 1:i32
    // CHECK-NEXT:   %b5:i10:unit = store %b5:i9:i32 %l1:i32*
    // CHECK-NEXT:   j b4()
  }

  // CHECK-LABEL: block b6:
  // CHECK-NEXT:   %b6:i0:i32 = load %l3:i32*
  // CHECK-NEXT:   ret %b6:i0:i32
  return t2;
}
// CHECK-NEXT: }

// CHECK: fun i32 @main () {
// CHECK-NEXT: init:
// CHECK-NEXT:   bid: b0
// CHECK-NEXT:   allocations:
int main() { 
  // CHECK-LABEL: block b0:
  // CHECK-NEXT:   %b0:i0:i32 = call @fibonacci:[ret:i32 params:(i32)]*(9:i32)
  // CHECK-NEXT:   %b0:i1:i1 = cmp eq %b0:i0:i32 34:i32
  // CHECK-NEXT:   %b0:i2:i32 = typecast %b0:i1:i1 to i32
  // CHECK-NEXT:   ret %b0:i2:i32
  return fibonacci(9) == 34; 
}
