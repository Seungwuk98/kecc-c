// RUN: kecc %s -S -emit-kecc -print-stdout | FileCheck %s
// clang-format off

// CHECK: fun i32 @main () {
// CHECK-NEXT: init:
// CHECK-NEXT:   bid: b0
// CHECK-NEXT:   allocations:
// CHECK-NEXT:     %l0:i32:y
// CHECK-NEXT:     %l1:i32:x
int main() {
  // CHECK-LABEL: block b0:
  int y = 2;
  // CHECK-NEXT:   %b0:i0:unit = store 2:i32 %l0:i32*

  int x = (y += 2, 2, y + 3);
  // CHECK-NEXT:   %b0:i1:i32 = load %l0:i32*
  // CHECK-NEXT:   %b0:i2:i32 = add %b0:i1:i32 2:i32
  // CHECK-NEXT:   %b0:i3:unit = store %b0:i2:i32 %l0:i32*
  // CHECK-NEXT:   %b0:i4:i32 = load %l0:i32*
  // CHECK-NEXT:   %b0:i5:i32 = add %b0:i4:i32 3:i32
  // CHECK-NEXT:   %b0:i6:unit = store %b0:i5:i32 %l1:i32*
  return x == 7;
  // CHECK-NEXT:   %b0:i7:i32 = load %l1:i32*
  // CHECK-NEXT:   %b0:i8:i1 = cmp eq %b0:i7:i32 7:i32
  // CHECK-NEXT:   %b0:i9:i32 = typecast %b0:i8:i1 to i32
  // CHECK-NEXT:   ret %b0:i9:i32
}
