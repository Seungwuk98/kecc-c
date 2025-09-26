// RUN: kecc %s -S -emit-kecc -print-stdout | FileCheck %s
// clang-format off

// CHECK: fun i32 @fibonacci (i32) {
// CHECK-NEXT: init:
// CHECK-NEXT:   bid: b0
// CHECK-NEXT:   allocations:
// CHECK-NEXT:     %l0:i32:n
int(fibonacci)(int n) {
  // CHECK-LABEL: block b0:
  // CHECK-NEXT:   %b0:p0:i32:n
  // CHECK-NEXT:   %b0:i0:unit = store %b0:p0:i32 %l0:i32*
  // CHECK-NEXT:   %b0:i1:i32 = load %l0:i32*
  // CHECK-NEXT:   %b0:i2:i1 = cmp lt %b0:i1:i32 2:i32
  // CHECK-NEXT:   %b0:i3:i32 = typecast %b0:i2:i1 to i32
  // CHECK-NEXT:   %b0:i4:i1 = cmp ne %b0:i3:i32 0:i32
  // CHECK-NEXT:   br %b0:i4:i1, b1(), b2()
  if (n < 2) {
    // CHECK-LABEL: block b1:
    // CHECK-NEXT:   %b1:i0:i32 = load %l0:i32*
    // CHECK-NEXT:   ret %b1:i0:i32
    return n;
  }
  // CHECK-LABEL: block b2:
  // CHECK-NEXT:   j b3()

  // CHECK-LABEL: block b3:
  // CHECK-NEXT:   %b3:i0:i32 = load %l0:i32*
  // CHECK-NEXT:   %b3:i1:i32 = sub %b3:i0:i32 2:i32
  // CHECK-NEXT:   %b3:i2:i32 = call @fibonacci:[ret:i32 params:(i32)]*(%b3:i1:i32)
  // CHECK-NEXT:   %b3:i3:i32 = load %l0:i32*
  // CHECK-NEXT:   %b3:i4:i32 = sub %b3:i3:i32 1:i32
  // CHECK-NEXT:   %b3:i5:i32 = call @fibonacci:[ret:i32 params:(i32)]*(%b3:i4:i32)
  // CHECK-NEXT:   %b3:i6:i32 = add %b3:i2:i32 %b3:i5:i32
  // CHECK-NEXT:   ret %b3:i6:i32
  return fibonacci(n - 2) + fibonacci(n - 1);
}

// CHECK: fun i32 @main () {
// CHECK-NEXT: init:
// CHECK-NEXT:   bid: b0
// CHECK-NEXT:   allocations:
int main() { 
  // CHECK-LABEL: block b0:
  // CHECK-NEXT:   %b0:i0:i32 = call @fibonacci:[ret:i32 params:(i32)]*(9:i32)
  // CHECK-NEXT:   %b0:i1:i1 = cmp eq %b0:i0:i32 34:i32
  // CHECK-NEXT:   %b0:i2:i32 = typecast %b0:i1:i1 to i32
  // CHECK-NEXT:   ret %b0:i2:i32
  return fibonacci(9) == 34; 
}
// CHECK-NEXT: }
