// RUN: kecc %s -S -emit-kecc -print-stdout | FileCheck %s
// RUN: keci %s --test-return-value=0
// clang-format off

// CHECK: fun i32 @int_greater_than (i32, u32) {
// CHECK-NEXT: init:
// CHECK-NEXT:   bid: b0
// CHECK-NEXT:   allocations:
// CHECK-NEXT:     %l0:i32:i
// CHECK-NEXT:     %l1:u32:j
int int_greater_than(int i, unsigned int j) {
  // CHECK-LABEL: block b0:
  // CHECK-NEXT:   %b0:p0:i32:i
  // CHECK-NEXT:   %b0:p1:u32:j
  // CHECK-NEXT:   %b0:i0:unit = store %b0:p0:i32 %l0:i32*
  // CHECK-NEXT:   %b0:i1:unit = store %b0:p1:u32 %l1:u32*
  // CHECK-NEXT:   %b0:i2:i32 = load %l0:i32*
  // CHECK-NEXT:   %b0:i3:u32 = typecast %b0:i2:i32 to u32
  // CHECK-NEXT:   %b0:i4:u32 = load %l1:u32*
  // CHECK-NEXT:   %b0:i5:i1 = cmp gt %b0:i3:u32 %b0:i4:u32
  // CHECK-NEXT:   %b0:i6:i32 = typecast %b0:i5:i1 to i32
  // CHECK-NEXT:   %b0:i7:i1 = cmp ne %b0:i6:i32 0:i32
  // CHECK-NEXT:   br %b0:i7:i1, b1(), b2()
  if (i > j)
    // CHECK-LABEL: block b1:
    // CHECK-NEXT:   ret 1:i32
    return 1;
  else
    // CHECK-LABEL: block b2:
    // CHECK-NEXT:   ret 0:i32
    return 0;
  // CHECK-LABEL: block b3:
  // CHECK-NEXT:   ret undef:i32
}
// CHECK-NEXT: }

// CHECK: fun i32 @char_greater_than (u8, u8) {
// CHECK-NEXT: init:
// CHECK-NEXT:   bid: b0
// CHECK-NEXT:   allocations:
// CHECK-NEXT:     %l0:u8:i
// CHECK-NEXT:     %l1:u8:j
int char_greater_than(char i, unsigned char j) {
  // CHECK-LABEL: block b0:
  // CHECK-NEXT:   %b0:p0:u8:i
  // CHECK-NEXT:   %b0:p1:u8:j
  // CHECK-NEXT:   %b0:i0:unit = store %b0:p0:u8 %l0:u8*
  // CHECK-NEXT:   %b0:i1:unit = store %b0:p1:u8 %l1:u8*
  // CHECK-NEXT:   %b0:i2:u8 = load %l0:u8*
  // CHECK-NEXT:   %b0:i3:i32 = typecast %b0:i2:u8 to i32
  // CHECK-NEXT:   %b0:i4:u8 = load %l1:u8*
  // CHECK-NEXT:   %b0:i5:i32 = typecast %b0:i4:u8 to i32
  // CHECK-NEXT:   %b0:i6:i1 = cmp gt %b0:i3:i32 %b0:i5:i32
  // CHECK-NEXT:   %b0:i7:i32 = typecast %b0:i6:i1 to i32
  // CHECK-NEXT:   %b0:i8:i1 = cmp ne %b0:i7:i32 0:i32
  // CHECK-NEXT:   br %b0:i8:i1, b1(), b2()
  if (i > j)
    // CHECK-LABEL: block b1:
    // CHECK-NEXT:   ret 1:i32
    return 1;
  else
    // CHECK-LABEL: block b2:
    // CHECK-NEXT:   ret 0:i32
    return 0;
  // CHECK-LABEL: block b3:
  // CHECK-NEXT:   ret undef:i32
}
// CHECK-NEXT: }

// CHECK: fun i32 @main () {
// CHECK-NEXT: init:
// CHECK-NEXT:   bid: b0
// CHECK-NEXT:   allocations:
// CHECK-NEXT:     %l0:i32:r1
// CHECK-NEXT:     %l1:i32:r2
// CHECK-NEXT:     %l2:i1
int main() {
  // cmp ugt
  // CHECK-LABEL: block b0:
  // CHECK-NEXT:   %b0:i0:i32 = minus 1:i32
  // CHECK-NEXT:   %b0:i1:i32 = call @int_greater_than:[ret:i32 params:(i32, u32)]*(%b0:i0:i32, 1:u32)
  // CHECK-NEXT:   %b0:i2:unit = store %b0:i1:i32 %l0:i32*
  int r1 = int_greater_than(-1, 1);
  // cmp sgt
  // CHECK-NEXT:   %b0:i3:i32 = minus 1:i32
  // CHECK-NEXT:   %b0:i4:u8 = typecast %b0:i3:i32 to u8
  // CHECK-NEXT:   %b0:i5:i32 = call @char_greater_than:[ret:i32 params:(u8, u8)]*(%b0:i4:u8, 1:u8)
  // CHECK-NEXT:   %b0:i6:unit = store %b0:i5:i32 %l1:i32*
  int r2 = char_greater_than(-1, 1);

  // CHECK-NEXT:   %b0:i7:i32 = load %l0:i32*
  // CHECK-NEXT:   %b0:i8:i1 = cmp eq %b0:i7:i32 1:i32
  // CHECK-NEXT:   %b0:i9:i32 = typecast %b0:i8:i1 to i32
  // CHECK-NEXT:   %b0:i10:i1 = cmp ne %b0:i9:i32 0:i32
  // CHECK-NEXT:   br %b0:i10:i1, b1(), b2()

  // CHECK-LABEL: block b1:
  // CHECK-NEXT:   %b1:i0:i32 = load %l1:i32*
  // CHECK-NEXT:   %b1:i1:i1 = cmp eq %b1:i0:i32 0:i32
  // CHECK-NEXT:   %b1:i2:i32 = typecast %b1:i1:i1 to i32
  // CHECK-NEXT:   %b1:i3:i1 = cmp ne %b1:i2:i32 0:i32
  // CHECK-NEXT:   %b1:i4:unit = store %b1:i3:i1 %l2:i1*
  // CHECK-NEXT:   j b3()

  // CHECK-LABEL: block b2:
  // CHECK-NEXT:   %b2:i0:unit = store 0:i1 %l2:i1*
  // CHECK-NEXT:   j b3()

  // CHECK-LABEL: block b3:
  // CHECK-NEXT:   %b3:i0:i1 = load %l2:i1*
  // CHECK-NEXT:   %b3:i1:i32 = typecast %b3:i0:i1 to i32
  // CHECK-NEXT:   ret %b3:i1:i32
  return r1 == 1 && r2 == 0;
}
// CHECK-NEXT: }
