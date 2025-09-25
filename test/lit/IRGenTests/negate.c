// RUN: kecc %s -S -emit-kecc -print-stdout | FileCheck %s
// clang-format off

// CHECK: fun i32 @foo (i32, i32, i32) {
// CHECK-NEXT: init:
// CHECK-NEXT:   bid: b0
// CHECK-NEXT:   allocations:
// CHECK-NEXT:     %l0:i32:x
// CHECK-NEXT:     %l1:i32:y
// CHECK-NEXT:     %l2:i32:z
int foo(int x, int y, int z) {
  // CHECK-LABEL: block b0:
  // CHECK-NEXT:   %b0:p0:i32:x
  // CHECK-NEXT:   %b0:p1:i32:y
  // CHECK-NEXT:   %b0:p2:i32:z
  // CHECK-NEXT:   %b0:i0:unit = store %b0:p0:i32 %l0:i32*
  // CHECK-NEXT:   %b0:i1:unit = store %b0:p1:i32 %l1:i32*
  // CHECK-NEXT:   %b0:i2:unit = store %b0:p2:i32 %l2:i32*
  
  // CHECK-NEXT:   %b0:i3:i32 = load %l0:i32*
  // CHECK-NEXT:   %b0:i4:i32 = load %l1:i32*
  // CHECK-NEXT:   %b0:i5:i1 = cmp eq %b0:i3:i32 %b0:i4:i32
  // CHECK-NEXT:   %b0:i6:i32 = typecast %b0:i5:i1 to i32
  // CHECK-NEXT:   %b0:i7:i1 = cmp ne %b0:i6:i32 0:i32
  // CHECK-NEXT:   %b0:i8:i1 = xor %b0:i7:i1 1:i1
  // CHECK-NEXT:   br %b0:i8:i1, b1(), b2()
  if (!(x == y)) {
    // CHECK-LABEL: block b1:
    // CHECK-NEXT:   %b1:i0:i32 = load %l1:i32*
    // CHECK-NEXT:   ret %b1:i0:i32
    return y;
  } else {
    // CHECK-LABEL: block b2:
    // CHECK-NEXT:   %b2:i0:i32 = load %l2:i32*
    // CHECK-NEXT:   ret %b2:i0:i32
    return z;
  }
  // CHECK-LABEL: block b3:
  // CHECK-NEXT:   ret undef:i32
}

// CHECK: fun i32 @main () {
// CHECK-NEXT: init:
// CHECK-NEXT:   bid: b0
// CHECK-NEXT:   allocations:
int main() { 
  // CHECK-LABEL: block b0:
  // CHECK-NEXT:   %b0:i0:i32 = minus 1:i32
  // CHECK-NEXT:   %b0:i1:i32 = call @foo:[ret:i32 params:(i32, i32, i32)]*(0:i32, 1:i32, %b0:i0:i32)
  // CHECK-NEXT:   %b0:i2:i1 = cmp eq %b0:i1:i32 1:i32
  // CHECK-NEXT:   %b0:i3:i32 = typecast %b0:i2:i1 to i32
  // CHECK-NEXT:   ret %b0:i3:i32
  return foo(0, 1, -1) == 1; 
}
// CHECK-NEXT: }
