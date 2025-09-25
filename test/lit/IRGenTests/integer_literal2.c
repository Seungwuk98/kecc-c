// RUN: kecc %s -S -emit-kecc -print-stdout | FileCheck %s
// clang-format off

// CHECK: fun i32 @main () {
// CHECK-NEXT: init:
// CHECK-NEXT:   bid: b0
// CHECK-NEXT:   allocations:
// CHECK-NEXT:     %l0:i32:temp
int main() {
  // CHECK-LABEL: block b0:
  int temp = 0;
  // CHECK-NEXT:   %b0:i0:unit = store 0:i32 %l0:i32*
  // `0xFFFFFFFF` is translated as `unsigned int` not `int`
  return temp < 0xFFFFFFFF;
  // CHECK-NEXT:   %b0:i1:i32 = load %l0:i32*
  // CHECK-NEXT:   %b0:i2:u32 = typecast %b0:i1:i32 to u32
  // CHECK-NEXT:   %b0:i3:i1 = cmp lt %b0:i2:u32 4294967295:u32
  // CHECK-NEXT:   %b0:i4:i32 = typecast %b0:i3:i1 to i32
  // CHECK-NEXT:   ret %b0:i4:i32
}
// CHECK-NEXT: }
