// RUN: kecc %s -S -emit-kecc -print-stdout | FileCheck %s
// clang-format off

// CHECK: fun i32 @main () {
// CHECK-NEXT: init:
// CHECK-NEXT:   bid: b0
// CHECK-NEXT:   allocations:
// CHECK-NEXT:     %l0:i32:i
// CHECK-NEXT:     %l1:i32:sum
int main() {
  // CHECK-LABEL: block b0:
  int i;
  int sum = 0;
  // CHECK-NEXT:   %b0:i0:unit = store 0:i32 %l1:i32*
  // CHECK-NEXT:   %b0:i1:unit = store 0:i32 %l0:i32*
  for (i = 0; i < 11; ++i) {
    // CHECK-LABEL: block b1:
    // CHECK-NEXT:   %b1:i0:i32 = load %l0:i32*
    // CHECK-NEXT:   %b1:i1:i1 = cmp lt %b1:i0:i32 11:i32
    // CHECK-NEXT:   %b1:i2:i32 = typecast %b1:i1:i1 to i32
    // CHECK-NEXT:   %b1:i3:i1 = cmp ne %b1:i2:i32 0:i32
    // CHECK-NEXT:   br %b1:i3:i1, b2(), b4()

    // CHECK-LABEL: block b2:
    // CHECK-NEXT:   %b2:i0:i32 = load %l1:i32*
    // CHECK-NEXT:   %b2:i1:i32 = load %l0:i32*
    // CHECK-NEXT:   %b2:i2:i32 = add %b2:i0:i32 %b2:i1:i32
    // CHECK-NEXT:   %b2:i3:unit = store %b2:i2:i32 %l1:i32*
    // CHECK-NEXT:   j b3()
    sum += i;

    // CHECK-LABEL: block b3:
    // CHECK-NEXT:   %b3:i0:i32 = load %l0:i32*
    // CHECK-NEXT:   %b3:i1:i32 = add %b3:i0:i32 1:i32
    // CHECK-NEXT:   %b3:i2:unit = store %b3:i1:i32 %l0:i32*
    // CHECK-NEXT:   j b1()
  }

  // CHECK-LABEL: block b4:
  // CHECK-NEXT:   %b4:i0:i32 = load %l1:i32*
  // CHECK-NEXT:   %b4:i1:i1 = cmp eq %b4:i0:i32 55:i32
  // CHECK-NEXT:   %b4:i2:i32 = typecast %b4:i1:i1 to i32
  // CHECK-NEXT:   ret %b4:i2:i32
  return sum == 55;
}
// CHECK-NEXT: }
