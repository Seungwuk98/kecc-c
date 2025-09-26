// RUN: kecc %s -S -emit-kecc -print-stdout | FileCheck %s
// clang-format off

int nonce = 1; // For random input
// CHECK-LABEL: var i32 @nonce = 1

// CHECK: fun i32 @foo () {
// CHECK-NEXT: init:
// CHECK-NEXT:   bid: b0
// CHECK-NEXT:   allocations:
// CHECK-NEXT:    %l0:i32:sum
// CHECK-NEXT:    %l1:i32:i 
// CHECK-NEXT:    %l2:i32:continue_num 
int foo() {
  // CHECK-LABEL: block b0:
  int sum = 0;
  // CHECK-NEXT:   %b0:i0:unit = store 0:i32 %l0:i32*
  int i = 0;
  // CHECK-NEXT:   %b0:i1:unit = store 0:i32 %l1:i32*
  int continue_num = nonce % 98;
  // CHECK-NEXT:   %b0:i2:i32 = load @nonce:i32*
  // CHECK-NEXT:   %b0:i3:i32 = mod %b0:i2:i32 98:i32
  // CHECK-NEXT:   %b0:i4:unit = store %b0:i3:i32 %l2:i32*
  // CHECK-NEXT:   j b1()

  while (i < 100) {
    // CHECK-LABEL: block b1:
    // CHECK-NEXT:   %b1:i0:i32 = load %l1:i32*
    // CHECK-NEXT:   %b1:i1:i1 = cmp lt %b1:i0:i32 100:i32
    // CHECK-NEXT:   %b1:i2:i32 = typecast %b1:i1:i1 to i32
    // CHECK-NEXT:   %b1:i3:i1 = cmp ne %b1:i2:i32 0:i32
    // CHECK-NEXT:   br %b1:i3:i1, b2(), b3()
   
    // CHECK-LABEL: block b2:
    // CHECK-NEXT:   %b2:i0:i32 = load %l1:i32*
    // CHECK-NEXT:   %b2:i1:i32 = load %l2:i32*
    // CHECK-NEXT:   %b2:i2:i1 = cmp eq %b2:i0:i32 %b2:i1:i32
    // CHECK-NEXT:   %b2:i3:i32 = typecast %b2:i2:i1 to i32
    // CHECK-NEXT:   %b2:i4:i1 = cmp ne %b2:i3:i32 0:i32
    // CHECK-NEXT:   br %b2:i4:i1, b4(), b5()

    // CHECK-LABEL: block b3:
    // CHECK-NEXT:   %b3:i0:i32 = load %l0:i32*
    // CHECK-NEXT:   ret %b3:i0:i32
    if (i == continue_num) {
      // CHECK-LABEL: block b4:
      // CHECK-NEXT:   %b4:i0:i32 = load %l1:i32*
      // CHECK-NEXT:   %b4:i1:i32 = add %b4:i0:i32 1:i32
      // CHECK-NEXT:   %b4:i2:unit = store %b4:i1:i32 %l1:i32*
      // CHECK-NEXT:   j b1()
      i++;
      continue;
    }
    // CHECK-LABEL: block b5:
    // CHECK-NEXT:   j b6()

    // CHECK-LABEL: block b6:
    sum += i;
    // CHECK-NEXT:   %b6:i0:i32 = load %l0:i32*
    // CHECK-NEXT:   %b6:i1:i32 = load %l1:i32*
    // CHECK-NEXT:   %b6:i2:i32 = add %b6:i0:i32 %b6:i1:i32
    // CHECK-NEXT:   %b6:i3:unit = store %b6:i2:i32 %l0:i32*
 
    i++;
    // CHECK-NEXT:   %b6:i4:i32 = load %l1:i32*
    // CHECK-NEXT:   %b6:i5:i32 = add %b6:i4:i32 1:i32
    // CHECK-NEXT:   %b6:i6:unit = store %b6:i5:i32 %l1:i32*

    // CHECK-NEXT:   %b6:i7:i32 = load %l1:i32*
    // CHECK-NEXT:   %b6:i8:i32 = load %l2:i32*
    // CHECK-NEXT:   %b6:i9:i32 = add %b6:i8:i32 2:i32
    // CHECK-NEXT:   %b6:i10:i1 = cmp eq %b6:i7:i32 %b6:i9:i32
    // CHECK-NEXT:   %b6:i11:i32 = typecast %b6:i10:i1 to i32
    // CHECK-NEXT:   %b6:i12:i1 = cmp ne %b6:i11:i32 0:i32
    // CHECK-NEXT:   br %b6:i12:i1, b7(), b8()
    if (i == continue_num + 2)
      // CHECK-LABEL: block b7:
      // CHECK-NEXT:   j b3()
      break;

    // CHECK-LABEL: block b8:
    // CHECK-NEXT:   j b9()

    // CHECK-LABEL: block b9:
    // CHECK-NEXT:   j b1()
  }

  return sum;
}
// CHECK-NEXT: }

// CHECK: fun i32 @main () {
// CHECK-NEXT: init:
// CHECK-NEXT:   bid: b0
// CHECK-NEXT:   allocations:
int main() { 
  // CHECK-LABEL: block b0:
  // CHECK-NEXT:   %b0:i0:i32 = call @foo:[ret:i32 params:()]*()
  // CHECK-NEXT:   ret %b0:i0:i32
  return foo(); 
}
// CHECK-NEXT: }
