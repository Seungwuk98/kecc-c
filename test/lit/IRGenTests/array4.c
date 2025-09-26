// RUN: kecc %s -S -emit-kecc -print-stdout | FileCheck %s
// clang-format off

// CHECK: fun i32 @main () { 
// CHECK-NEXT: init:
// CHECK-NEXT:   bid: b0
// CHECK-NEXT:   allocations:
// CHECK-NEXT:     %l0:[10 x i32]:a 
// CHECK-NEXT:     %l1:i32*:p
// CHECK-NEXT:     %l2:i32:i
int main() {
  // CHECK-LABEL: block b0:
  int a[10];
  int *p = a;
  // CHECK-NEXT:   %b0:i0:i32* = getelementptr %l0:[10 x i32]* offset 0:i64
  // CHECK-NEXT:   %b0:i1:unit = store %b0:i0:i32* %l1:i32**

  // CHECK-NEXT:   %b0:i2:unit = store 0:i32 %l2:i32*
  // CHECK-NEXT:   j b1()
  for (int i = 0; i < 10; i++) {
    // CHECK-LABEL: block b1:
    // CHECK-NEXT:   %b1:i0:i32 = load %l2:i32*
    // CHECK-NEXT:   %b1:i1:i1 = cmp lt %b1:i0:i32 10:i32
    // CHECK-NEXT:   %b1:i2:i32 = typecast %b1:i1:i1 to i32
    // CHECK-NEXT:   %b1:i3:i1 = cmp ne %b1:i2:i32 0:i32
    // CHECK-NEXT:   br %b1:i3:i1, b2(), b4()

    // CHECK-LABEL: block b2:
    // CHECK-NEXT:   %b2:i0:i32* = load %l1:i32**
    // CHECK-NEXT:   %b2:i1:i32* = getelementptr %b2:i0:i32* offset 4:i64
    // CHECK-NEXT:   %b2:i2:unit = store %b2:i1:i32* %l1:i32**
    // CHECK-NEXT:   %b2:i3:i32 = load %l2:i32*
    // CHECK-NEXT:   %b2:i4:unit = store %b2:i3:i32 %b2:i0:i32*
    // CHECK-NEXT:   j b3()
    *(p++) = i;

    // CHECK-LABEL: block b3:
    // CHECK-NEXT:   %b3:i0:i32 = load %l2:i32*
    // CHECK-NEXT:   %b3:i1:i32 = add %b3:i0:i32 1:i32
    // CHECK-NEXT:   %b3:i2:unit = store %b3:i1:i32 %l2:i32*
    // CHECK-NEXT:   j b1()
  }

  // CHECK-LABEL: block b4:
  // CHECK-NEXT:   %b4:i0:i32* = getelementptr %l0:[10 x i32]* offset 0:i64
  // CHECK-NEXT:   %b4:i1:i64 = typecast 5:i32 to i64
  // CHECK-NEXT:   %b4:i2:i64 = mul %b4:i1:i64 4:i64
  // CHECK-NEXT:   %b4:i3:i32* = getelementptr %b4:i0:i32* offset %b4:i2:i64
  // CHECK-NEXT:   %b4:i4:i32 = load %b4:i3:i32*
  // CHECK-NEXT:   %b4:i5:i1 = cmp eq %b4:i4:i32 5:i32
  // CHECK-NEXT:   %b4:i6:i32 = typecast %b4:i5:i1 to i32
  // CHECK-NEXT:   ret %b4:i6:i32
  return a[5] == 5;
}
// CHECK-NEXT: }
