// RUN: kecc %s -S -emit-kecc -print-stdout | FileCheck %s
// clang-format off

// CHECK: fun i32 @main () {
// CHECK-NEXT: init:
// CHECK-NEXT:   bid: b0
// CHECK-NEXT:   allocations:
// CHECK-NEXT:     %l0:i32:a
// CHECK-NEXT:     %l1:i32:b 
// CHECK-NEXT:     %l2:i32:c 
// CHECK-NEXT:     %l3:i32:d 
// CHECK-NEXT:     %l4:i1
// CHECK-NEXT:     %l5:i1
// CHECK-NEXT:     %l6:i1
int main() {
  // CHECK-LABEL: block b0:
  int a = 0;
  // CHECK-NEXT:   %b0:i0:unit = store 0:i32 %l0:i32*
  int b = 0;
  // CHECK-NEXT:   %b0:i1:unit = store 0:i32 %l1:i32*
  int c = 0;
  // CHECK-NEXT:   %b0:i2:unit = store 0:i32 %l2:i32*
  int d = 0;
  // CHECK-NEXT:   %b0:i3:unit = store 0:i32 %l3:i32*

  // CHECK-NEXT:   %b0:i4:unit = store 1:i32 %l0:i32*
  // CHECK-NEXT:   %b0:i5:i1 = cmp ne 1:i32 0:i32
  // CHECK-NEXT:   br %b0:i5:i1, b1(), b2()

  // CHECK-LABEL: block b1:
  // CHECK-NEXT:   %b1:i0:unit = store 1:i1 %l4:i1*
  // CHECK-NEXT:   j b3()

  // CHECK-LABEL: block b2:
  // CHECK-NEXT:   %b2:i0:unit = store 1:i32 %l1:i32*
  // CHECK-NEXT:   %b2:i1:i1 = cmp ne 1:i32 0:i32
  // CHECK-NEXT:   %b2:i2:unit = store %b2:i1:i1 %l4:i1*
  // CHECK-NEXT:   j b3()

  // CHECK-LABEL: block b3:
  // CHECK-NEXT:   %b3:i0:i1 = load %l4:i1*
  // CHECK-NEXT:   br %b3:i0:i1, b4(), b5()
  if ((a = 1) || (b = 1)) {
    // CHECK-LABEL: block b4:
    // CHECK-NEXT:   %b4:i0:i32 = load %l1:i32*
    // CHECK-NEXT:   %b4:i1:i32 = add %b4:i0:i32 1:i32
    // CHECK-NEXT:   %b4:i2:unit = store %b4:i1:i32 %l1:i32*
    // CHECK-NEXT:   j b6()
    b++;
  }
  // CHECK-LABEL: block b5:
  // CHECK-NEXT:   j b6()

  // CHECK-LABEL: block b6:
  // CHECK-NEXT:   %b6:i0:unit = store 1:i32 %l2:i32*
  // CHECK-NEXT:   %b6:i1:i1 = cmp ne 1:i32 0:i32
  // CHECK-NEXT:   br %b6:i1:i1, b7(), b8()

  // CHECK-LABEL: block b7:
  // CHECK-NEXT:   %b7:i0:unit = store 1:i32 %l3:i32*
  // CHECK-NEXT:   %b7:i1:i1 = cmp ne 1:i32 0:i32
  // CHECK-NEXT:   %b7:i2:unit = store %b7:i1:i1 %l5:i1*
  // CHECK-NEXT:   j b9()

  // CHECK-LABEL: block b8:
  // CHECK-NEXT:   %b8:i0:unit = store 0:i1 %l5:i1*
  // CHECK-NEXT:   j b9()

  // CHECK-LABEL: block b9:
  // CHECK-NEXT:   %b9:i0:i1 = load %l5:i1*
  // CHECK-NEXT:   br %b9:i0:i1, b10(), b11()
  if ((c = 1) && (d = 1)) {
    // CHECK-LABEL: block b10:
    // CHECK-NEXT:   %b10:i0:i32 = load %l3:i32*
    // CHECK-NEXT:   %b10:i1:i32 = add %b10:i0:i32 1:i32
    // CHECK-NEXT:   %b10:i2:unit = store %b10:i1:i32 %l3:i32*
    // CHECK-NEXT:   j b12()
    d++;
  }
  // CHECK-LABEL: block b11:
  // CHECK-NEXT:   j b12()

  // CHECK-LABEL: block b12:
  // CHECK-NEXT:   %b12:i0:i32 = load %l1:i32*
  // CHECK-NEXT:   %b12:i1:i1 = cmp eq %b12:i0:i32 1:i32
  // CHECK-NEXT:   %b12:i2:i32 = typecast %b12:i1:i1 to i32
  // CHECK-NEXT:   %b12:i3:i1 = cmp ne %b12:i2:i32 0:i32
  // CHECK-NEXT:   br %b12:i3:i1, b13(), b14()

  // CHECK-LABEL: block b13:
  // CHECK-NEXT:   %b13:i0:i32 = load %l3:i32*
  // CHECK-NEXT:   %b13:i1:i1 = cmp eq %b13:i0:i32 2:i32
  // CHECK-NEXT:   %b13:i2:i32 = typecast %b13:i1:i1 to i32
  // CHECK-NEXT:   %b13:i3:i1 = cmp ne %b13:i2:i32 0:i32
  // CHECK-NEXT:   %b13:i4:unit = store %b13:i3:i1 %l6:i1*
  // CHECK-NEXT:   j b15()

  // CHECK-LABEL: block b14:
  // CHECK-NEXT:   %b14:i0:unit = store 0:i1 %l6:i1*
  // CHECK-NEXT:   j b15()

  // CHECK-LABEL: block b15:
  // CHECK-NEXT:   %b15:i0:i1 = load %l6:i1*
  // CHECK-NEXT:   %b15:i1:i32 = typecast %b15:i0:i1 to i32
  // CHECK-NEXT:   ret %b15:i1:i32
  return b == 1 && d == 2;
}
// CHECK-NEXT: }
