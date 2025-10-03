// RUN: kecc %s -S -emit-kecc -print-stdout | FileCheck %s
// RUN: keci %s --test-return-value=1
// clang-format off

// CHECK: fun i32 @main () {
// CHECK-NEXT: init:
// CHECK-NEXT:   bid: b0
// CHECK-NEXT:   allocations:
// CHECK-NEXT:    %l0:i64:l
// CHECK-NEXT:    %l1:i64:l2
// CHECK-NEXT:    %l2:i64:l3
// CHECK-NEXT:    %l3:i16:s
// CHECK-NEXT:    %l4:i16:s2
// CHECK-NEXT:    %l5:i32:i
// CHECK-NEXT:    %l6:u8:c
int main() {
  // CHECK-LABEL: block b0:
  long int l = 1;
  // CHECK-NEXT:   %b0:i0:unit = store 1:i64 %l0:i64*
  long l2 = 2;
  // CHECK-NEXT:   %b0:i1:unit = store 2:i64 %l1:i64*
  long long l3 = 3;
  // CHECK-NEXT:   %b0:i2:unit = store 3:i64 %l2:i64*
  short int s = 4;
  // CHECK-NEXT:   %b0:i3:unit = store 4:i16 %l3:i16*
  short s2 = 5;
  // CHECK-NEXT:   %b0:i4:unit = store 5:i16 %l4:i16*
  int i = 6;
  // CHECK-NEXT:   %b0:i5:unit = store 6:i32 %l5:i32*
  char c = 7;
  // CHECK-NEXT:   %b0:i6:unit = store 7:u8 %l6:u8*

  // CHECK-NEXT:   %b0:i7:i64 = load %l0:i64*
  // CHECK-NEXT:   %b0:i8:i64 = load %l1:i64*
  // CHECK-NEXT:   %b0:i9:i64 = add %b0:i7:i64 %b0:i8:i64
  // CHECK-NEXT:   %b0:i10:i64 = load %l2:i64*
  // CHECK-NEXT:   %b0:i11:i64 = add %b0:i9:i64 %b0:i10:i64
  // CHECK-NEXT:   %b0:i12:i16 = load %l3:i16*
  // CHECK-NEXT:   %b0:i13:i64 = typecast %b0:i12:i16 to i64
  // CHECK-NEXT:   %b0:i14:i64 = add %b0:i11:i64 %b0:i13:i64
  // CHECK-NEXT:   %b0:i15:i16 = load %l4:i16*
  // CHECK-NEXT:   %b0:i16:i64 = typecast %b0:i15:i16 to i64
  // CHECK-NEXT:   %b0:i17:i64 = add %b0:i14:i64 %b0:i16:i64
  // CHECK-NEXT:   %b0:i18:i32 = load %l5:i32*
  // CHECK-NEXT:   %b0:i19:i64 = typecast %b0:i18:i32 to i64
  // CHECK-NEXT:   %b0:i20:i64 = add %b0:i17:i64 %b0:i19:i64
  // CHECK-NEXT:   %b0:i21:u8 = load %l6:u8*
  // CHECK-NEXT:   %b0:i22:i64 = typecast %b0:i21:u8 to i64
  // CHECK-NEXT:   %b0:i23:i64 = add %b0:i20:i64 %b0:i22:i64
  // CHECK-NEXT:   %b0:i24:i1 = cmp eq %b0:i23:i64 28:i64
  // CHECK-NEXT:   %b0:i25:i32 = typecast %b0:i24:i1 to i32
  // CHECK-NEXT:   ret %b0:i25:i32
  return (l + l2 + l3 + s + s2 + i + c) == 28;
}
// CHECK-NEXT: }
