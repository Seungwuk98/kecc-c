// RUN: kecc %s -S -emit-kecc -print-stdout | FileCheck %s
// clang-format off

// CHECK: fun i32 @main () {
// CHECK-NEXT: init:
// CHECK-NEXT:   bid: b0
// CHECK-NEXT:   allocations:
// CHECK-NEXT:     %l0:i32:i 
// CHECK-NEXT:     %l1:i32:i
// CHECK-NEXT:     %l2:i32:i
// CHECK-NEXT:     %l3:i32:k 
int main() {
  // CHECK-LABEL: block b0: 
  int i = 0;
  // CHECK-NEXT:   %b0:i0:unit = store 0:i32 %l0:i32* 

  // CHECK-NEXT:   %b0:i1:unit = store 0:i32 %l1:i32*
  // CHECK-NEXT:   j b1()
  for (int i = 0; i < 10; ++i) {
    // CHECK-LABEL: block b1: 
    // CHECK-NEXT:   %b1:i0:i32 = load %l1:i32*
    // CHECK-NEXT:   %b1:i1:i1 = cmp lt %b1:i0:i32 10:i32
    // CHECK-NEXT:   %b1:i2:i32 = typecast %b1:i1:i1 to i32
    // CHECK-NEXT:   %b1:i3:i1 = cmp ne %b1:i2:i32 0:i32
    // CHECK-NEXT:   br %b1:i3:i1, b2(), b4()

    // CHECK-LABEL: block b2:
    // CHECK-NEXT:   %b2:i0:unit = store 0:i32 %l2:i32*
    // CHECK-NEXT:   %b2:i1:unit = store 0:i32 %l3:i32*
    // CHECK-NEXT:   j b3()
    int i = 0;
    int k = 0;
    // CHECK-LABEL: block b3:
    // CHECK-NEXT:   %b3:i0:i32 = load %l1:i32*
    // CHECK-NEXT:   %b3:i1:i32 = add %b3:i0:i32 1:i32
    // CHECK-NEXT:   %b3:i2:unit = store %b3:i1:i32 %l1:i32*
    // CHECK-NEXT:   j b1()
  }

  // CHECK-LABEL: block b4:
  // CHECK-NEXT:   ret 1:i32
  return 1;
}
// CHECK-NEXT: }
