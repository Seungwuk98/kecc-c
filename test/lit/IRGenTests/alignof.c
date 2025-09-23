// RUN: kecc %s -S -emit-kecc -print-stdout | FileCheck %s

int main() {
  // CHECK: fun i32 @main () {
  // CHECK-NEXT: init:
  // CHECK-NEXT:   bid: b0
  // CHECK-NEXT:   allocations:

  // CHECK-LABEL: block b0:
  // CHECK-NEXT:   %b0:i0:i1 = cmp eq 4:u32 4:u32
  // CHECK-NEXT:   %b0:i1:i32 = typecast %b0:i0:i1 to i32
  // CHECK-NEXT:   ret %b0:i1:i32
  return _Alignof(const int) == 4;
}
