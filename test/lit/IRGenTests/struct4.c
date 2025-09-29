// RUN: kecc %s -S -emit-kecc -print-stdout | FileCheck %s
// RUN: keci %s --test-return-value=1
// clang-format off

// CHECK-DAG: struct Foo : { x:i32 }
// CHECK-DAG: var i32 @nonce = 1
int nonce = 1; // For random input

struct Foo {
  int x;
};

// CHECK: fun struct Foo @f () {
// CHECK-NEXT: init:
// CHECK-NEXT:   bid: b0
// CHECK-NEXT:   allocations:
// CHECK-NEXT:    %l0:struct Foo:x
struct Foo f() {
  // CHECK-LABEL: block b0:
  struct Foo x;
  x.x = nonce;
  // CHECK-NEXT:   %b0:i0:i32* = getelementptr %l0:struct Foo* offset 0:i64
  // CHECK-NEXT:   %b0:i1:i32 = load @nonce:i32*
  // CHECK-NEXT:   %b0:i2:unit = store %b0:i1:i32 %b0:i0:i32*
  // CHECK-NEXT:   %b0:i3:struct Foo = load %l0:struct Foo*
  // CHECK-NEXT:   ret %b0:i3:struct Foo
  return x;
}
// CHECK-NEXT: }

// CHECK: fun i32 @main () {
// CHECK-NEXT: init:
// CHECK-NEXT:   bid: b0
// CHECK-NEXT:   allocations:
// CHECK-NEXT:    %l0:i32:x
// CHECK-NEXT:    %l1:struct Foo
int main() {
  // CHECK-LABEL: block b0:
  // CHECK-NEXT:   %b0:i0:struct Foo = call @f:[ret:struct Foo params:()]*()
  // CHECK-NEXT:   %b0:i1:unit = store %b0:i0:struct Foo %l1:struct Foo*
  // CHECK-NEXT:   %b0:i2:i32* = getelementptr %l1:struct Foo* offset 0:i64
  // CHECK-NEXT:   %b0:i3:i32 = load %b0:i2:i32*
  // CHECK-NEXT:   %b0:i4:unit = store %b0:i3:i32 %l0:i32*
  int x = f().x;
  // CHECK-NEXT:   %b0:i5:i32 = load %l0:i32*
  // CHECK-NEXT:   ret %b0:i5:i32
  return x;
}
// CHECK-NEXT: }
