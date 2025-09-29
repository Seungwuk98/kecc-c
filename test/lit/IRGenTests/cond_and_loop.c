// RUN: kecc %s -S -emit-kecc -print-stdout | FileCheck %s
// RUN: keci %s --test-return-value=9
// clang-format off

int nonce = 1; // For random input
// CHECK: var i32 @nonce = 1

// CHECK: fun i32 @main () {
// CHECK-NEXT: init:
// CHECK-NEXT:   bid: b0
// CHECK-NEXT:   allocations:
// CHECK-NEXT:     %l0:i32:i
// CHECK-NEXT:     %l1:i32:p 
// CHECK-NEXT:     %l2:i32:q 
// CHECK-NEXT:     %l3:i32:r
// CHECK-NEXT:     %l4:i32
// CHECK-NEXT:     %l5:i32
// CHECK-NEXT:     %l6:i32:loop_num 
// CHECK-NEXT:     %l7:i32
int main() {
  // CHECK-LABEL: block b0:
  int i;
  int p = 2;
  // CHECK-NEXT:   %b0:i0:unit = store 2:i32 %l1:i32*
  int q = 5;
  // CHECK-NEXT:   %b0:i1:unit = store 5:i32 %l2:i32*
  int r = (0 ? ((p > q) ? (p -= 2) : (p += 2)) : (p + q));
  // CHECK-NEXT:   %b0:i2:i1 = cmp ne 0:i32 0:i32
  // CHECK-NEXT:   br %b0:i2:i1, b1(), b2()

  // CHECK-LABEL: block b1:
  // CHECK-NEXT:   %b1:i0:i32 = load %l1:i32*
  // CHECK-NEXT:   %b1:i1:i32 = load %l2:i32*
  // CHECK-NEXT:   %b1:i2:i1 = cmp gt %b1:i0:i32 %b1:i1:i32 
  // CHECK-NEXT:   %b1:i3:i32 = typecast %b1:i2:i1 to i32
  // CHECK-NEXT:   %b1:i4:i1 = cmp ne %b1:i3:i32 0:i32
  // CHECK-NEXT:   br %b1:i4:i1, b4(), b5()

  // CHECK-LABEL: block b2:
  // CHECK-NEXT:   %b2:i0:i32 = load %l1:i32*
  // CHECK-NEXT:   %b2:i1:i32 = load %l2:i32*
  // CHECK-NEXT:   %b2:i2:i32 = add %b2:i0:i32 %b2:i1:i32
  // CHECK-NEXT:   %b2:i3:unit = store %b2:i2:i32 %l4:i32*
  // CHECK-NEXT:   j b3()

  // CHECK-LABEL: block b3:
  // CHECK-NEXT:   %b3:i0:i32 = load %l4:i32*
  // CHECK-NEXT:   %b3:i1:unit = store %b3:i0:i32 %l3:i32*
  // CHECK-NEXT:   %b3:i2:i32 = load @nonce:i32*
  // CHECK-NEXT:   %b3:i3:i32 = mod %b3:i2:i32 100:i32
  // CHECK-NEXT:   %b3:i4:unit = store %b3:i3:i32 %l6:i32*
  // CHECK-NEXT:   %b3:i5:unit = store 0:i32 %l0:i32*
  // CHECK-NEXT:   j b7()

  // CHECK-LABEL: block b4:
  // CHECK-NEXT:   %b4:i0:i32 = load %l1:i32*
  // CHECK-NEXT:   %b4:i1:i32 = sub %b4:i0:i32 2:i32
  // CHECK-NEXT:   %b4:i2:unit = store %b4:i1:i32 %l1:i32*
  // CHECK-NEXT:   %b4:i3:unit = store %b4:i1:i32 %l5:i32*
  // CHECK-NEXT:   j b6()

  // CHECK-LABEL: block b5:
  // CHECK-NEXT:   %b5:i0:i32 = load %l1:i32*
  // CHECK-NEXT:   %b5:i1:i32 = add %b5:i0:i32 2:i32
  // CHECK-NEXT:   %b5:i2:unit = store %b5:i1:i32 %l1:i32*
  // CHECK-NEXT:   %b5:i3:unit = store %b5:i1:i32 %l5:i32*
  // CHECK-NEXT:   j b6()

  // CHECK-LABEL: block b6:
  // CHECK-NEXT:   %b6:i0:i32 = load %l5:i32*
  // CHECK-NEXT:   %b6:i1:unit = store %b6:i0:i32 %l4:i32*
  // CHECK-NEXT:   j b3()
  int loop_num = nonce % 100;

  for (i = 0; i < loop_num; ((i % 2) ? (i += 2) : ++i)) {
    // CHECK-LABEL: block b7:
    // CHECK-NEXT:   %b7:i0:i32 = load %l0:i32*
    // CHECK-NEXT:   %b7:i1:i32 = load %l6:i32*
    // CHECK-NEXT:   %b7:i2:i1 = cmp lt %b7:i0:i32 %b7:i1:i32
    // CHECK-NEXT:   %b7:i3:i32 = typecast %b7:i2:i1 to i32
    // CHECK-NEXT:   %b7:i4:i1 = cmp ne %b7:i3:i32 0:i32
    // CHECK-NEXT:   br %b7:i4:i1, b8(), b10()

    // CHECK-LABEL: block b8:
    // CHECK-NEXT:   %b8:i0:i32 = load %l0:i32*
    // CHECK-NEXT:   %b8:i1:i32 = mod %b8:i0:i32 2:i32
    // CHECK-NEXT:   %b8:i2:i1 = cmp ne %b8:i1:i32 0:i32
    // CHECK-NEXT:   br %b8:i2:i1, b11(), b12()

    // CHECK-LABEL: block b9:
    // CHECK-NEXT:   %b9:i0:i32 = load %l0:i32*
    // CHECK-NEXT:   %b9:i1:i32 = mod %b9:i0:i32 2:i32
    // CHECK-NEXT:   %b9:i2:i1 = cmp ne %b9:i1:i32 0:i32
    // CHECK-NEXT:   br %b9:i2:i1, b14(), b15()

    // CHECK-LABEL: block b10:
    // CHECK-NEXT:   %b10:i0:i32 = load %l1:i32*
    if (i % 2) {
      // CHECK-LABEL: block b11:
      // CHECK-NEXT:   %b11:i0:i32 = load %l1:i32*
      // CHECK-NEXT:   %b11:i1:i32 = load %l2:i32*
      // CHECK-NEXT:   %b11:i2:i32 = add %b11:i0:i32 %b11:i1:i32
      // CHECK-NEXT:   %b11:i3:unit = store %b11:i2:i32 %l1:i32*
      // CHECK-NEXT:   j b13()
      p += q;
    } else {
      // CHECK-LABEL: block b12:
      // CHECK-NEXT:   %b12:i0:i32 = load %l1:i32*
      // CHECK-NEXT:   %b12:i1:i32 = load %l3:i32*
      // CHECK-NEXT:   %b12:i2:i32 = add %b12:i0:i32 %b12:i1:i32
      // CHECK-NEXT:   %b12:i3:unit = store %b12:i2:i32 %l1:i32*
      // CHECK-NEXT:   j b13()
      p += r;
    }
    // CHECK-LABEL: block b13:
    // CHECK-NEXT:   j b9()

    // CHECK-LABEL: block b14:
    // CHECK-NEXT:   %b14:i0:i32 = load %l0:i32*
    // CHECK-NEXT:   %b14:i1:i32 = add %b14:i0:i32 2:i32
    // CHECK-NEXT:   %b14:i2:unit = store %b14:i1:i32 %l0:i32*
    // CHECK-NEXT:   %b14:i3:unit = store %b14:i1:i32 %l7:i32*
    // CHECK-NEXT:   j b16()

    // CHECK-LABEL: block b15:
    // CHECK-NEXT:   %b15:i0:i32 = load %l0:i32*
    // CHECK-NEXT:   %b15:i1:i32 = add %b15:i0:i32 1:i32
    // CHECK-NEXT:   %b15:i2:unit = store %b15:i1:i32 %l0:i32*
    // CHECK-NEXT:   %b15:i3:unit = store %b15:i1:i32 %l7:i32*
    // CHECK-NEXT:   j b16()
    
    // CHECK-LABEL: block b16:
    // CHECK-NEXT:   %b16:i0:i32 = load %l7:i32*
    // CHECK-NEXT:   j b7()
  }
  
  return p;
}
