// RUN: kecc %s -S -emit-kecc -print-stdout | FileCheck %s
// RUN: keci %s --test-return-value=255
// clang-format off

unsigned int crc32_context = 0xFFFFFFFFUL;
// CHECK: var u32 @crc32_context = 4294967295

// CHECK: fun i32 @main () {
// CHECK-NEXT: init:
// CHECK-NEXT:   bid: b0
// CHECK-NEXT:   allocations:
int main() { 
  // CHECK-LABEL: block b0:
  // CHECK-NEXT:   %b0:i0:u32 = load @crc32_context:u32*
  // CHECK-NEXT:   %b0:i1:u8 = typecast %b0:i0:u32 to u8
  // CHECK-NEXT:   %b0:i2:i32 = typecast %b0:i1:u8 to i32
  // CHECK-NEXT:   ret %b0:i2:i32
  return (unsigned char)crc32_context; 
}
// CHECK-NEXT: }
