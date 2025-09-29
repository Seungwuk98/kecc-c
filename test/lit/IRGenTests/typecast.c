// RUN: kecc %s -S -emit-kecc -print-stdout | FileCheck %s
// RUN: keci %s --test-return-value=1
// clang-format off

char temp = 0x00L;
// CHECK-LABEL: var u8 @temp = 0

// CHECK: fun i32 @main () {
// CHECK-NEXT: init:
// CHECK-NEXT:   bid: b0
// CHECK-NEXT:   allocations:
int main() { 
  // CHECK-LABEL: block b0:
  // CHECK-NEXT:   %b0:i0:unit = store 54:u8 @temp:u8*
  // CHECK-NEXT:   %b0:i1:i1 = cmp ge 54:i32 2:i32
  // CHECK-NEXT:   %b0:i2:i32 = typecast %b0:i1:i1 to i32
  // CHECK-NEXT:   ret %b0:i2:i32
  return (temp = 0xEF36L) >= (2L); 
}
// CHECK-NEXT: }
