// RUN: kecc %s -S -emit-kecc -print-stdout | FileCheck %s
// clang-format off

// CHECK: fun i32 @main () {
// CHECK-NEXT: init:
// CHECK-NEXT:   bid: b0
// CHECK-NEXT:   allocations:
// CHECK-NEXT:    %l0:i32:i 
// CHECK-NEXT:    %l1:i32:c 
int main() {
  // CHECK-LABEL: block b0:
  int i = 0;
  // CHECK-NEXT:   %b0:i0:unit = store 0:i32 %l0:i32*
  int c = 0;
  // CHECK-NEXT:   %b0:i1:unit = store 0:i32 %l1:i32*
  // CHECK-NEXT:   j b1()
  while (i < 10) {
    // CHECK-LABEL: block b1:
    // CHECK-NEXT:   %b1:i0:i32 = load %l0:i32*
    // CHECK-NEXT:   %b1:i1:i1 = cmp lt %b1:i0:i32 10:i32
    // CHECK-NEXT:   %b1:i2:i32 = typecast %b1:i1:i1 to i32
    // CHECK-NEXT:   %b1:i3:i1 = cmp ne %b1:i2:i32 0:i32
    // CHECK-NEXT:   br %b1:i3:i1, b2(), b3()

    // CHECK-LABEL: block b2:
    i++;
    // CHECK-NEXT:   %b2:i0:i32 = load %l0:i32*
    // CHECK-NEXT:   %b2:i1:i32 = add %b2:i0:i32 1:i32
    // CHECK-NEXT:   %b2:i2:unit = store %b2:i1:i32 %l0:i32*
    // CHECK-NEXT:   %b2:i3:i32 = load %l0:i32*
    // CHECK-NEXT:   switch %b2:i3:i32 default b4() [
    // CHECK-NEXT:     1:i32 b6()
    // CHECK-NEXT:   ]

    // CHECK-LABEL: block b3:
    // CHECK-NEXT:   %b3:i0:i32 = load %l1:i32*
    // CHECK-NEXT:   ret %b3:i0:i32

    // CHECK-LABEL: block b4:
    // CHECK-NEXT:   j b5()

    // CHECK-LABEL: block b5:
    // CHECK-NEXT:   %b5:i0:i32 = load %l1:i32*
    // CHECK-NEXT:   %b5:i1:i32 = add %b5:i0:i32 1:i32
    // CHECK-NEXT:   %b5:i2:unit = store %b5:i1:i32 %l1:i32*
    // CHECK-NEXT:   j b1()

    // CHECK-LABEL: block b6:
    // CHECK-NEXT:   j b1()
    switch (i) {
    case (1): {
      continue;
      break;
    }
    default: {
      break;
    }
    }
    c++;
  }
  return c;
}
// CHECK-NEXT: }
