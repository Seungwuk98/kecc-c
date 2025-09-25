// RUN: kecc %s -S -emit-kecc -print-stdout | FileCheck %s
// clang-format off

// CHECK: fun i32* @foo (i32*) {
// CHECK-NEXT: init:
// CHECK-NEXT:   bid: b0
// CHECK-NEXT:   allocations:
// CHECK-NEXT:     %l0:i32*:a
int *foo(int *a) { 
  // CHECK-LABEL: block b0:
  // CHECK-NEXT:   %b0:p0:i32*:a
  // CHECK-NEXT:   %b0:i0:unit = store %b0:p0:i32* %l0:i32**
  // CHECK-NEXT:   %b0:i1:i32* = load %l0:i32**
  // CHECK-NEXT:   ret %b0:i1:i32*
  return a; 
}
// CHECK-NEXT: }

// CHECK: fun i32 @main () {
// CHECK-NEXT: init:
// CHECK-NEXT:   bid: b0
// CHECK-NEXT:   allocations:
// CHECK-NEXT:     %l0:i32:a 
// CHECK-NEXT:     %l1:i32*:p 
// CHECK-NEXT:     %l2:i32**:p2
// CHECK-NEXT:     %l3:i32*:p3
int main() {
  // CHECK-LABEL: block b0:
  int a = 1;
  // CHECK-NEXT:   %b0:i0:unit = store 1:i32 %l0:i32*
  int *p = &a;
  // CHECK-NEXT:   %b0:i1:unit = store %l0:i32* %l1:i32**
  int **p2 = &*&p;
  // CHECK-NEXT:   %b0:i2:unit = store %l1:i32** %l2:i32***
  int *p3 = *&p;
  // CHECK-NEXT:   %b0:i3:i32* = load %l1:i32**
  // CHECK-NEXT:   %b0:i4:unit = store %b0:i3:i32* %l3:i32**

  *&*foo(*p2) += 1;
  // CHECK-NEXT:   %b0:i5:i32** = load %l2:i32***
  // CHECK-NEXT:   %b0:i6:i32* = load %b0:i5:i32**
  // CHECK-NEXT:   %b0:i7:i32* = call @foo:[ret:i32* params:(i32*)]*(%b0:i6:i32*)
  // CHECK-NEXT:   %b0:i8:i32 = load %b0:i7:i32*
  // CHECK-NEXT:   %b0:i9:i32 = add %b0:i8:i32 1:i32
  // CHECK-NEXT:   %b0:i10:unit = store %b0:i9:i32 %b0:i7:i32*

  *foo(p3) += 1;
  // CHECK-NEXT:   %b0:i11:i32* = load %l3:i32**
  // CHECK-NEXT:   %b0:i12:i32* = call @foo:[ret:i32* params:(i32*)]*(%b0:i11:i32*)
  // CHECK-NEXT:   %b0:i13:i32 = load %b0:i12:i32*
  // CHECK-NEXT:   %b0:i14:i32 = add %b0:i13:i32 1:i32
  // CHECK-NEXT:   %b0:i15:unit = store %b0:i14:i32 %b0:i12:i32*

  // CHECK-NEXT:   %b0:i16:i32 = load %l0:i32*
  // CHECK-NEXT:   %b0:i17:i1 = cmp eq %b0:i16:i32 3:i32
  // CHECK-NEXT:   %b0:i18:i32 = typecast %b0:i17:i1 to i32
  // CHECK-NEXT:   ret %b0:i18:i32
  return a == 3;
}
// CHECK-NEXT: }
