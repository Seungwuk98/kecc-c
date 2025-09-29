// RUN: kecc %s -S -emit-kecc -print-stdout | FileCheck %s
// RUN: keci %s --test-return-value=1
// clang-format off

// CHECK: fun i32 @foo () {
// CHECK-NEXT: init:
// CHECK-NEXT:   bid: b0
// CHECK-NEXT:   allocations:
// CHECK-NEXT:     %l0:i32:sum
// CHECK-NEXT:     %l1:i32:i
int foo() {
  // CHECK-LABEL: block b0:
  int sum = 0;
  // CHECK-NEXT:   %b0:i0:unit = store 0:i32 %l0:i32*

  // CHECK-NEXT:   %b0:i1:unit = store 0:i32 %l1:i32*
  // CHECK-NEXT:   j b1()
  for (int i = 0;;) {
    // CHECK-LABEL: block b1:
    // CHECK-NEXT:   j b2()

    // CHECK-LABEL: block b2:
    // CHECK-NEXT:   %b2:i0:i32 = load %l1:i32*
    // CHECK-NEXT:   %b2:i1:i1 = cmp eq %b2:i0:i32 5:i32
    // CHECK-NEXT:   %b2:i2:i32 = typecast %b2:i1:i1 to i32
    // CHECK-NEXT:   %b2:i3:i1 = cmp ne %b2:i2:i32 0:i32
    // CHECK-NEXT:   br %b2:i3:i1, b5(), b6()

    // CHECK-LABEL: block b3:
    // CHECK-NEXT:   j b1()

    // CHECK-LABEL: block b4:
    // CHECK-NEXT:   %b4:i0:i32 = load %l0:i32*
    // CHECK-NEXT:   ret %b4:i0:i32

    if (i == 5)
      // CHECK-LABEL: block b5:
      // CHECK-NEXT:   j b4()
      break;
    // CHECK-LABEL: block b6:
    // CHECK-NEXT:   j b7()
    
    // CHECK-LABEL: block b7:
    // CHECK-NEXT:   %b7:i0:i32 = load %l1:i32*
    // CHECK-NEXT:   %b7:i1:i1 = cmp eq %b7:i0:i32 3:i32
    // CHECK-NEXT:   %b7:i2:i32 = typecast %b7:i1:i1 to i32
    // CHECK-NEXT:   %b7:i3:i1 = cmp ne %b7:i2:i32 0:i32
    // CHECK-NEXT:   br %b7:i3:i1, b8(), b9()
    if (i == 3) {
      // CHECK-LABEL: block b8:
      // CHECK-NEXT:   %b8:i0:i32 = load %l1:i32*
      // CHECK-NEXT:   %b8:i1:i32 = add %b8:i0:i32 1:i32
      // CHECK-NEXT:   %b8:i2:unit = store %b8:i1:i32 %l1:i32*
      // CHECK-NEXT:   j b3()
      i++;
      continue;
    }
    // CHECK-LABEL: block b9:
    // CHECK-NEXT:   j b10()

    // CHECK-LABEL: block b10:
    // CHECK-NEXT:   %b10:i0:i32 = load %l0:i32*
    // CHECK-NEXT:   %b10:i1:i32 = load %l1:i32*
    // CHECK-NEXT:   %b10:i2:i32 = add %b10:i0:i32 %b10:i1:i32
    // CHECK-NEXT:   %b10:i3:unit = store %b10:i2:i32 %l0:i32*
    // CHECK-NEXT:   %b10:i4:i32 = load %l1:i32*
    // CHECK-NEXT:   %b10:i5:i32 = add %b10:i4:i32 1:i32
    // CHECK-NEXT:   %b10:i6:unit = store %b10:i5:i32 %l1:i32*
    // CHECK-NEXT:   j b3()
    sum += i;
    i++;
  }
  return sum;
}

// CHECK: fun i32 @main () {
// CHECK-NEXT: init:
// CHECK-NEXT:   bid: b0 
// CHECK-NEXT:   allocations: 
int main() { 
  // CHECK-LABEL: block b0:
  // CHECK-NEXT:   %b0:i0:i32 = call @foo:[ret:i32 params:()]*()
  // CHECK-NEXT:   %b0:i1:i1 = cmp eq %b0:i0:i32 7:i32
  // CHECK-NEXT:   %b0:i2:i32 = typecast %b0:i1:i1 to i32
  // CHECK-NEXT:   ret %b0:i2:i32
  return foo() == 7; 
}
