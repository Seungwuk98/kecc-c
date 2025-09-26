// RUN: kecc %s -S -emit-kecc -print-stdout | FileCheck %s
// clang-format off

// CHECK: fun i32 @main () {
// CHECK-NEXT: init:
// CHECK-NEXT:   bid: b0
// CHECK-NEXT:   allocations:
// CHECK-NEXT:     %l0:u8:a
// CHECK-NEXT:     %l1:u8:b
// CHECK-NEXT:     %l2:[10 x i32]:c
// CHECK-NEXT:     %l3:i1
// CHECK-NEXT:     %l4:i1
int main() {
  // CHECK-LABEL: block b0:
  char a = 42, b = 5;
  // CHECK-NEXT:   %b0:i0:unit = store 42:u8 %l0:u8*
  // CHECK-NEXT:   %b0:i1:unit = store 5:u8 %l1:u8*
  long c[10];

  // CHECK-NEXT:   %b0:i2:i1 = cmp eq 1:u32 1:u32
  // CHECK-NEXT:   %b0:i3:i32 = typecast %b0:i2:i1 to i32
  // CHECK-NEXT:   %b0:i4:i1 = cmp ne %b0:i3:i32 0:i32
  // CHECK-NEXT:   br %b0:i4:i1, b1(), b2()

  // CHECK-LABEL: block b1:
  // CHECK-NEXT:   %b1:i0:i1 = cmp eq 4:u32 4:u32
  // CHECK-NEXT:   %b1:i1:i32 = typecast %b1:i0:i1 to i32
  // CHECK-NEXT:   %b1:i2:i1 = cmp ne %b1:i1:i32 0:i32
  // CHECK-NEXT:   %b1:i3:unit = store %b1:i2:i1 %l3:i1*
  // CHECK-NEXT:   j b3()

  // CHECK-LABEL: block b2:
  // CHECK-NEXT:   %b2:i0:unit = store 0:i1 %l3:i1*
  // CHECK-NEXT:   j b3()

  // CHECK-LABEL: block b3:
  // CHECK-NEXT:   %b3:i0:i1 = load %l3:i1*
  // CHECK-NEXT:   br %b3:i0:i1, b4(), b5()

  // CHECK-LABEL: block b4:
  // CHECK-NEXT:   %b4:i0:i1 = cmp eq 40:u32 80:u32 
  // CHECK-NEXT:   %b4:i1:i32 = typecast %b4:i0:i1 to i32
  // CHECK-NEXT:   %b4:i2:i1 = cmp ne %b4:i1:i32 0:i32
  // CHECK-NEXT:   %b4:i3:unit = store %b4:i2:i1 %l4:i1*
  // CHECK-NEXT:   j b6()

  // CHECK-LABEL: block b5:
  // CHECK-NEXT:   %b5:i0:unit = store 0:i1 %l4:i1*
  // CHECK-NEXT:   j b6()

  // CHECK-LABEL: block b6:
  // CHECK-NEXT:   %b6:i0:i1 = load %l4:i1*
  // CHECK-NEXT:   %b6:i1:i32 = typecast %b6:i0:i1 to i32
  // CHECK-NEXT:   ret %b6:i1:i32
  return sizeof(a) == 1 && sizeof(a + b) == 4 && sizeof(c) == 80;
}
// CHECK-NEXT: }
