// RUN: kecc %s -S -emit-kecc -print-stdout | FileCheck %s
// RUN: keci %s --test-return-value=1
// clang-format off

int a = -1;
// CHECK: var i32 @a = -1
long b = -1l;
// CHECK: var i32 @b = -1L
float c = -1.5f;
// CHECK: var f32 @c = -1.5f
double d = -1.5;
// CHECK: var f64 @d = -1.5

// CHECK: fun i32 @main () {
// CHECK-NEXT: init:
// CHECK-NEXT:   bid: b0
// CHECK-NEXT:   allocations:
int main() { 
  // CHECK-LABEL: block b0:
  // CHECK-NEXT:   %b0:i0:i32 = load @a:i32*
  // CHECK-NEXT:   %b0:i1:i32 = load @b:i32*
  // CHECK-NEXT:   %b0:i2:i32 = add %b0:i0:i32 %b0:i1:i32
  // CHECK-NEXT:   %b0:i3:f32 = load @c:f32*
  // CHECK-NEXT:   %b0:i4:i32 = typecast %b0:i3:f32 to i32
  // CHECK-NEXT:   %b0:i5:i32 = add %b0:i2:i32 %b0:i4:i32
  // CHECK-NEXT:   %b0:i6:f64 = load @d:f64*
  // CHECK-NEXT:   %b0:i7:i32 = typecast %b0:i6:f64 to i32
  // CHECK-NEXT:   %b0:i8:i32 = add %b0:i5:i32 %b0:i7:i32
  // CHECK-NEXT:   %b0:i9:i32 = minus 4:i32
  // CHECK-NEXT:   %b0:i10:i1 = cmp eq %b0:i8:i32 %b0:i9:i32
  // CHECK-NEXT:   %b0:i11:i32 = typecast %b0:i10:i1 to i32
  // CHECK-NEXT:   ret %b0:i11:i32
  return (a + b + (int)c + (long)d) == -4; 
}
// CHECK-NEXT: }
