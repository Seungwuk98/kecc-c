// RUN: kecc %s -S -emit-kecc -print-stdout | FileCheck %s
// RUN: keci %s --test-return-value=11
// clang-format off

int g_a[5] = {1, 2, 3};
// CHECK: var [5 x i32] @g_a = {1, 2, 3}

// CHECK: fun i32 @main () {
// CHECK-NEXT: init:
// CHECK-NEXT:   bid: b0
// CHECK-NEXT:   allocations:
// CHECK-NEXT:     %l0:i32:init
// CHECK-NEXT:     %l1:[5 x i32]:a 
// CHECK-NEXT:     %l2:i32:sum 
// CHECK-NEXT:     %l3:i32:i
int main() {
  // CHECK-LABEL: block b0:
  // CHECK-NEXT:   %b0:i0:unit = store 1:i32 %l0:i32*
  // CHECK-NEXT:   %b0:i1:i32* = getelementptr %l1:[5 x i32]* offset 0:i64
  // CHECK-NEXT:   %b0:i2:i32 = load %l0:i32*
  // CHECK-NEXT:   %b0:i3:unit = store %b0:i2:i32 %b0:i1:i32*
  // CHECK-NEXT:   %b0:i4:i32* = getelementptr %l1:[5 x i32]* offset 4:i64
  // CHECK-NEXT:   %b0:i5:unit = store 2:i32 %b0:i4:i32*
  // CHECK-NEXT:   %b0:i6:i32* = getelementptr %l1:[5 x i32]* offset 8:i64
  // CHECK-NEXT:   %b0:i7:unit = store 3:i32 %b0:i6:i32*
  // CHECK-NEXT:   %b0:i8:i32* = getelementptr %l1:[5 x i32]* offset 12:i64
  // CHECK-NEXT:   %b0:i9:unit = store 4:i32 %b0:i8:i32*
  // CHECK-NEXT:   %b0:i10:i32* = getelementptr %l1:[5 x i32]* offset 16:i64
  // CHECK-NEXT:   %b0:i11:i32 = minus 5:i32
  // CHECK-NEXT:   %b0:i12:unit = store %b0:i11:i32 %b0:i10:i32*
  int init = 1;
  int a[5] = {init, 2, 3, 4, -5};
  int sum = 0;
  // CHECK-NEXT:   %b0:i13:unit = store 0:i32 %l2:i32*
  // CHECK-NEXT:   %b0:i14:unit = store 0:i32 %l3:i32*
  // CHECK-NEXT:   j b1()
  
  for (int i = 0; i < 5; i++) {
    // CHECK-LABEL: block b1:
    // CHECK-NEXT:   %b1:i0:i32 = load %l3:i32*
    // CHECK-NEXT:   %b1:i1:i1 = cmp lt %b1:i0:i32 5:i32
    // CHECK-NEXT:   %b1:i2:i32 = typecast %b1:i1:i1 to i32
    // CHECK-NEXT:   %b1:i3:i1 = cmp ne %b1:i2:i32 0:i32
    // CHECK-NEXT:   br %b1:i3:i1, b2(), b4()
    
    // CHECK-LABEL: block b2:
    sum += a[i];
    // CHECK-NEXT:   %b2:i0:i32 = load %l2:i32*
    // CHECK-NEXT:   %b2:i1:i32* = getelementptr %l1:[5 x i32]* offset 0:i64
    // CHECK-NEXT:   %b2:i2:i32 = load %l3:i32*
    // CHECK-NEXT:   %b2:i3:i64 = typecast %b2:i2:i32 to i64
    // CHECK-NEXT:   %b2:i4:i64 = mul %b2:i3:i64 4:i64
    // CHECK-NEXT:   %b2:i5:i32* = getelementptr %b2:i1:i32* offset %b2:i4:i64
    // CHECK-NEXT:   %b2:i6:i32 = load %b2:i5:i32*
    // CHECK-NEXT:   %b2:i7:i32 = add %b2:i0:i32 %b2:i6:i32
    // CHECK-NEXT:   %b2:i8:unit = store %b2:i7:i32 %l2:i32*
    sum += g_a[i];
    // CHECK-NEXT:   %b2:i9:i32 = load %l2:i32*
    // CHECK-NEXT:   %b2:i10:i32* = getelementptr @g_a:[5 x i32]* offset 0:i64
    // CHECK-NEXT:   %b2:i11:i32 = load %l3:i32*
    // CHECK-NEXT:   %b2:i12:i64 = typecast %b2:i11:i32 to i64
    // CHECK-NEXT:   %b2:i13:i64 = mul %b2:i12:i64 4:i64
    // CHECK-NEXT:   %b2:i14:i32* = getelementptr %b2:i10:i32* offset %b2:i13:i64
    // CHECK-NEXT:   %b2:i15:i32 = load %b2:i14:i32*
    // CHECK-NEXT:   %b2:i16:i32 = add %b2:i9:i32 %b2:i15:i32
    // CHECK-NEXT:   %b2:i17:unit = store %b2:i16:i32 %l2:i32*
    // CHECK-NEXT:   j b3()

    // CHECK-LABEL: block b3:
    // CHECK-NEXT:   %b3:i0:i32 = load %l3:i32*
    // CHECK-NEXT:   %b3:i1:i32 = add %b3:i0:i32 1:i32
    // CHECK-NEXT:   %b3:i2:unit = store %b3:i1:i32 %l3:i32*
    // CHECK-NEXT:   j b1()
  }

  // CHECK-LABEL: block b4:
  // CHECK-NEXT:   %b4:i0:i32 = load %l2:i32*
  // CHECK-NEXT:   ret %b4:i0:i32
  return sum;
}
// CHECK-NEXT: }
