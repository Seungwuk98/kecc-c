// RUN: kecc %s -S -emit-kecc -print-stdout | FileCheck %s
// clang-format off

// CHECK: fun i32 @gcd (i32, i32) {
// CHECK-NEXT: init:
// CHECK-NEXT:   bid: b0
// CHECK-NEXT:   allocations:
// CHECK-NEXT:     %l0:i32:a
// CHECK-NEXT:     %l1:i32:b 
// CHECK-NEXT:     %l2:i32
// CHECK-NEXT:     %l3:i32
int gcd(int a, int b) {
  // CHECK-LABEL: block b0:
  // CHECK-NEXT:   %b0:p0:i32:a
  // CHECK-NEXT:   %b0:p1:i32:b 
  // CHECK-NEXT:   %b0:i0:unit = store %b0:p0:i32 %l0:i32* 
  // CHECK-NEXT:   %b0:i1:unit = store %b0:p1:i32 %l1:i32*

  a = (a > 0) ? a : -a;
  // CHECK-NEXT:   %b0:i2:i32 = load %l0:i32*
  // CHECK-NEXT:   %b0:i3:i1 = cmp gt %b0:i2:i32 0:i32
  // CHECK-NEXT:   %b0:i4:i32 = typecast %b0:i3:i1 to i32
  // CHECK-NEXT:   %b0:i5:i1 = cmp ne %b0:i4:i32 0:i32
  // CHECK-NEXT:   br %b0:i5:i1, b1(), b2()

  // CHECK-LABEL: block b1:
  // CHECK-NEXT:   %b1:i0:i32 = load %l0:i32*
  // CHECK-NEXT:   %b1:i1:unit = store %b1:i0:i32 %l2:i32*
  // CHECK-NEXT:   j b3()

  // CHECK-LABEL: block b2:
  // CHECK-NEXT:   %b2:i0:i32 = load %l0:i32*
  // CHECK-NEXT:   %b2:i1:i32 = minus %b2:i0:i32
  // CHECK-NEXT:   %b2:i2:unit = store %b2:i1:i32 %l2:i32*
  // CHECK-NEXT:   j b3()

  // CHECK-LABEL: block b3:
  // CHECK-NEXT:   %b3:i0:i32 = load %l2:i32*
  // CHECK-NEXT:   %b3:i1:unit = store %b3:i0:i32 %l0:i32*

  b = (b > 0) ? b : -b;
  // CHECK-NEXT:   %b3:i2:i32 = load %l1:i32*
  // CHECK-NEXT:   %b3:i3:i1 = cmp gt %b3:i2:i32 0:i32
  // CHECK-NEXT:   %b3:i4:i32 = typecast %b3:i3:i1 to i32
  // CHECK-NEXT:   %b3:i5:i1 = cmp ne %b3:i4:i32 0:i32
  // CHECK-NEXT:   br %b3:i5:i1, b4(), b5()

  // CHECK-LABEL: block b4:
  // CHECK-NEXT:   %b4:i0:i32 = load %l1:i32*
  // CHECK-NEXT:   %b4:i1:unit = store %b4:i0:i32 %l3:i32*
  // CHECK-NEXT:   j b6()

  // CHECK-LABEL: block b5:
  // CHECK-NEXT:   %b5:i0:i32 = load %l1:i32*
  // CHECK-NEXT:   %b5:i1:i32 = minus %b5:i0:i32
  // CHECK-NEXT:   %b5:i2:unit = store %b5:i1:i32 %l3:i32*
  // CHECK-NEXT:   j b6()

  // CHECK-LABEL: block b6:
  // CHECK-NEXT:   %b6:i0:i32 = load %l3:i32*
  // CHECK-NEXT:   %b6:i1:unit = store %b6:i0:i32 %l1:i32*
  // CHECK-NEXT:   j b7()
  while (a != b) {
    // CHECK-LABEL: block b7:
    // CHECK-NEXT:   %b7:i0:i32 = load %l0:i32*
    // CHECK-NEXT:   %b7:i1:i32 = load %l1:i32*
    // CHECK-NEXT:   %b7:i2:i1 = cmp ne %b7:i0:i32 %b7:i1:i32
    // CHECK-NEXT:   %b7:i3:i32 = typecast %b7:i2:i1 to i32
    // CHECK-NEXT:   %b7:i4:i1 = cmp ne %b7:i3:i32 0:i32
    // CHECK-NEXT:   br %b7:i4:i1, b8(), b9()

    // CHECK-LABEL: block b8:
    // CHECK-NEXT:   %b8:i0:i32 = load %l0:i32*
    // CHECK-NEXT:   %b8:i1:i32 = load %l1:i32*
    // CHECK-NEXT:   %b8:i2:i1 = cmp gt %b8:i0:i32 %b8:i1:i32
    // CHECK-NEXT:   %b8:i3:i32 = typecast %b8:i2:i1 to i32
    // CHECK-NEXT:   %b8:i4:i1 = cmp ne %b8:i3:i32 0:i32
    // CHECK-NEXT:   br %b8:i4:i1, b10(), b11()

    // CHECK-LABEL: block b9:
    // CHECK-NEXT:   %b9:i0:i32 = load %l0:i32*
    // CHECK-NEXT:   ret %b9:i0:i32
    if (a > b) {
      // CHECK-LABEL: block b10:
      // CHECK-NEXT:   %b10:i0:i32 = load %l0:i32*
      // CHECK-NEXT:   %b10:i1:i32 = load %l1:i32*
      // CHECK-NEXT:   %b10:i2:i32 = sub %b10:i0:i32 %b10:i1:i32
      // CHECK-NEXT:   %b10:i3:unit = store %b10:i2:i32 %l0:i32*
      // CHECK-NEXT:   j b12()
      a -= b;
    } else {
      // CHECK-LABEL: block b11:
      // CHECK-NEXT:   %b11:i0:i32 = load %l1:i32*
      // CHECK-NEXT:   %b11:i1:i32 = load %l0:i32*
      // CHECK-NEXT:   %b11:i2:i32 = sub %b11:i0:i32 %b11:i1:i32
      // CHECK-NEXT:   %b11:i3:unit = store %b11:i2:i32 %l1:i32*
      // CHECK-NEXT:   j b12()
      b -= a;
    }
    // CHECK-LABEL: block b12:
    // CHECK-NEXT:   j b7()
  }
  return a;
}
// CHECK-NEXT: }

// CHECK: fun i32 @main () {
// CHECK-NEXT: init:
// CHECK-NEXT:   bid: b0 
// CHECK-NEXT:   allocations:
int main() { 
  // CHECK-LABEL: block b0:
  // CHECK-NEXT:   %b0:i0:i32 = call @gcd:[ret:i32 params:(i32, i32)]*(18:i32, 21:i32)
  // CHECK-NEXT:   %b0:i1:i1 = cmp eq %b0:i0:i32 3:i32
  // CHECK-NEXT:   %b0:i2:i32 = typecast %b0:i1:i1 to i32
  // CHECK-NEXT:   ret %b0:i2:i32
  return gcd(18, 21) == 3; 
}
// CHECK-NEXT: }
