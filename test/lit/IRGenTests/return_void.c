// RUN: kecc %s -S -emit-kecc -print-stdout | FileCheck %s
// clang-format off

// CHECK: fun unit @foo () {
// CHECK-NEXT: init:
// CHECK-NEXT:   bid: b0
// CHECK-NEXT:   allocations:
void foo() {
  // CHECK-LABEL: block b0:
  // CHECK-NEXT:   ret unit:unit
}
// CHECK-NEXT: }

// CHECK: fun i32 @main () {
// CHECK-NEXT: init:
// CHECK-NEXT:   bid: b0
// CHECK-NEXT:   allocations:
int main() {
  // CHECK-LABEL: block b0:
  foo();
  // CHECK-NEXT:   %b0:i0:unit = call @foo:[ret:unit params:()]*()
  return 1;
  // CHECK-NEXT:   ret 1:i32
}
// CHECK-NEXT: }
