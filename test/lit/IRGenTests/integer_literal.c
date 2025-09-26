// RUN: kecc %s -S -emit-kecc -print-stdout | FileCheck %s
// clang-format off

// CHECK: fun i32 @main () {
// CHECK-NEXT: init:
// CHECK-NEXT:   bid: b0
// CHECK-NEXT:   allocations:
// CHECK-NEXT:     %l0:i16:temp 
// CHECK-NEXT:     %l1:u32:temp2 
int main() {
  // CHECK-LABEL: block b0:
  short temp = 0;
  // CHECK-NEXT:   %b0:i0:unit = store 0:i16 %l0:i16*
  unsigned int temp2 = 4294967163;
  // CHECK-NEXT:   %b0:i1:unit = store 4294967163:u32 %l1:u32*
  return (char)(temp ^ temp2) == 123;
  // CHECK-NEXT:   %b0:i2:i16 = load %l0:i16*
  // CHECK-NEXT:   %b0:i3:u32 = typecast %b0:i2:i16 to u32
  // CHECK-NEXT:   %b0:i4:u32 = load %l1:u32*
  // CHECK-NEXT:   %b0:i5:u32 = xor %b0:i3:u32 %b0:i4:u32
  // CHECK-NEXT:   %b0:i6:u8 = typecast %b0:i5:u32 to u8
  // CHECK-NEXT:   %b0:i7:i32 = typecast %b0:i6:u8 to i32
  // CHECK-NEXT:   %b0:i8:i1 = cmp eq %b0:i7:i32 123:i32
  // CHECK-NEXT:   %b0:i9:i32 = typecast %b0:i8:i1 to i32
  // CHECK-NEXT:   ret %b0:i9:i32
}
// CHECK-NEXT: }
