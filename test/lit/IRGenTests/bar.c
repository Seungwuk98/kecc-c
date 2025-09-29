// RUN: kecc %s -S -emit-kecc -print-stdout | FileCheck %s
// RUN: keci %s --test-return-value=1
// clang-format off

// CHECK: fun i32 @bar (i32, i32, i32) {
// CHECK-NEXT: init:
// CHECK-NEXT:   bid: b0
// CHECK-NEXT:   allocations:
// CHECK-NEXT:     %l0:i32:x
// CHECK-NEXT:     %l1:i32:y
// CHECK-NEXT:     %l2:i32:z
// CHECK-NEXT:     %l3:i32:arith_mean
// CHECK-NEXT:     %l4:i32:ugly_mean
int bar(int x, int y, int z) {
  // CHECK-LABEL: block b0:
  // CHECK-NEXT:   %b0:p0:i32:x
  // CHECK-NEXT:   %b0:p1:i32:y
  // CHECK-NEXT:   %b0:p2:i32:z
  // CHECK-NEXT:   %b0:i0:unit = store %b0:p0:i32 %l0:i32*
  // CHECK-NEXT:   %b0:i1:unit = store %b0:p1:i32 %l1:i32*
  // CHECK-NEXT:   %b0:i2:unit = store %b0:p2:i32 %l2:i32*

  int arith_mean = (x + y + z) / 3;
  // CHECK-NEXT:   %b0:i3:i32 = load %l0:i32*
  // CHECK-NEXT:   %b0:i4:i32 = load %l1:i32*
  // CHECK-NEXT:   %b0:i5:i32 = add %b0:i3:i32 %b0:i4:i32
  // CHECK-NEXT:   %b0:i6:i32 = load %l2:i32*
  // CHECK-NEXT:   %b0:i7:i32 = add %b0:i5:i32 %b0:i6:i32
  // CHECK-NEXT:   %b0:i8:i32 = div %b0:i7:i32 3:i32
  // CHECK-NEXT:   %b0:i9:unit = store %b0:i8:i32 %l3:i32*

  int ugly_mean = (((x + y) / 2) * 2 + z) / 3;
  // CHECK-NEXT:   %b0:i10:i32 = load %l0:i32*
  // CHECK-NEXT:   %b0:i11:i32 = load %l1:i32*
  // CHECK-NEXT:   %b0:i12:i32 = add %b0:i10:i32 %b0:i11:i32
  // CHECK-NEXT:   %b0:i13:i32 = div %b0:i12:i32 2:i32
  // CHECK-NEXT:   %b0:i14:i32 = mul %b0:i13:i32 2:i32
  // CHECK-NEXT:   %b0:i15:i32 = load %l2:i32*
  // CHECK-NEXT:   %b0:i16:i32 = add %b0:i14:i32 %b0:i15:i32
  // CHECK-NEXT:   %b0:i17:i32 = div %b0:i16:i32 3:i32
  // CHECK-NEXT:   %b0:i18:unit = store %b0:i17:i32 %l4:i32*
  
  // CHECK-NEXT:   %b0:i19:i32 = load %l0:i32*
  // CHECK-NEXT:   %b0:i20:i32 = load %l1:i32*
  // CHECK-NEXT:   %b0:i21:i1 = cmp eq %b0:i19:i32 %b0:i20:i32
  // CHECK-NEXT:   %b0:i22:i32 = typecast %b0:i21:i1 to i32
  // CHECK-NEXT:   %b0:i23:i1 = cmp ne %b0:i22:i32 0:i32
  // CHECK-NEXT:   br %b0:i23:i1, b1(), b2()
  if (x == y) {
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

