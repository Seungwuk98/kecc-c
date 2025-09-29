// RUN: kecc %s -S -emit-kecc -print-stdout | FileCheck %s
// TODO: The output of cmp.c is 0 but this ir must be 1
//       There are difference between x86(=1) and riscv(=0)
// RUN: keci %s --test-return-value=0
// clang-format off

// CHECK: fun i32 @main () {
// CHECK-NEXT: init:
// CHECK-NEXT:   bid: b0
// CHECK-NEXT:   allocations:
// CHECK-NEXT:     %l0:u8:a
// CHECK-NEXT:     %l1:u8:b
// CHECK-NEXT:     %l2:u8:c
// CHECK-NEXT:     %l3:i1
int main() {
  // CHECK-LABEL: block b0:
  char a = 127;
  // CHECK-NEXT:   %b0:i0:unit = store 127:u8 %l0:u8*
  char b = a << 1;
  // CHECK-NEXT:   %b0:i1:u8 = load %l0:u8*
  // CHECK-NEXT:   %b0:i2:i32 = typecast %b0:i1:u8 to i32
  // CHECK-NEXT:   %b0:i3:i32 = shl %b0:i2:i32 1:i32
  // CHECK-NEXT:   %b0:i4:u8 = typecast %b0:i3:i32 to u8
  // CHECK-NEXT:   %b0:i5:unit = store %b0:i4:u8 %l1:u8*
  unsigned char c = (unsigned char)b >> 1;
  // CHECK-NEXT:   %b0:i6:u8 = load %l1:u8*
  // CHECK-NEXT:   %b0:i7:i32 = typecast %b0:i6:u8 to i32
  // CHECK-NEXT:   %b0:i8:i32 = shr %b0:i7:i32 1:i32
  // CHECK-NEXT:   %b0:i9:u8 = typecast %b0:i8:i32 to u8
  // CHECK-NEXT:   %b0:i10:unit = store %b0:i9:u8 %l2:u8*

  // CHECK-NEXT:   %b0:i11:u8 = load %l1:u8*
  // CHECK-NEXT:   %b0:i12:i32 = typecast %b0:i11:u8 to i32
  // CHECK-NEXT:   %b0:i13:i32 = minus 2:i32
  // CHECK-NEXT:   %b0:i14:i1 = cmp eq %b0:i12:i32 %b0:i13:i32
  // CHECK-NEXT:   %b0:i15:i32 = typecast %b0:i14:i1 to i32
  // CHECK-NEXT:   %b0:i16:i1 = cmp ne %b0:i15:i32 0:i32
  // CHECK-NEXT:   br %b0:i16:i1, b1(), b2()

  // CHECK-LABEL: block b1:
  // CHECK-NEXT:   %b1:i0:u8 = load %l2:u8*  
  // CHECK-NEXT:   %b1:i1:i32 = typecast %b1:i0:u8 to i32
  // CHECK-NEXT:   %b1:i2:i1 = cmp eq %b1:i1:i32 127:i32
  // CHECK-NEXT:   %b1:i3:i32 = typecast %b1:i2:i1 to i32
  // CHECK-NEXT:   %b1:i4:i1 = cmp ne %b1:i3:i32 0:i32
  // CHECK-NEXT:   %b1:i5:unit = store %b1:i4:i1 %l3:i1*
  // CHECK-NEXT:   j b3()

  // CHECK-LABEL: block b2:
  // CHECK-NEXT:   %b2:i0:unit = store 0:i1 %l3:i1* 
  // CHECK-NEXT:   j b3()

  // CHECK-LABEL: block b3:
  // CHECK-NEXT:   %b3:i0:i1 = load %l3:i1*
  // CHECK-NEXT:   %b3:i1:i32 = typecast %b3:i0:i1 to i32
  // CHECK-NEXT:   ret %b3:i1:i32
  return b == -2 && c == 0x7F;
}
// CHECK-NEXT: }

