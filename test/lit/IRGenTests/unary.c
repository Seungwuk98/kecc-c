// RUN: kecc %s -S -emit-kecc -print-stdout | FileCheck %s
// clang-format off

// CHECK: fun i32 @main () { 
// CHECK-NEXT: init:
// CHECK-NEXT:   bid: b0
// CHECK-NEXT:   allocations:
// CHECK-NEXT:    %l0:u8:temp
int main() {
  // CHECK-LABEL: block b0:
  unsigned char temp = 0x00L;
  // CHECK-NEXT:   %b0:i0:unit = store 0:u8 %l0:u8*
  // CHECK-NEXT:   %b0:i1:u8 = load %l0:u8*
  // CHECK-NEXT:   %b0:i2:u8 = sub %b0:i1:u8 1:u8
  // CHECK-NEXT:   %b0:i3:unit = store %b0:i2:u8 %l0:u8*
  // CHECK-NEXT:   %b0:i4:i32 = typecast %b0:i2:u8 to i32
  // CHECK-NEXT:   %b0:i5:i1 = cmp gt 1:i32 %b0:i4:i32
  // CHECK-NEXT:   %b0:i6:i32 = typecast %b0:i5:i1 to i32
  // CHECK-NEXT:   ret %b0:i6:i32
  return 1 > (--temp);
}
// CHECK-NEXT: }
