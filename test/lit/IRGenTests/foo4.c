// RUN: kecc %s -S -emit-kecc -print-stdout | FileCheck %s
// clang-format off

// CHECK: fun i32 @foo (i32, i32, i32) {
// CHECK-NEXT: init:
// CHECK-NEXT:   bid: b0
// CHECK-NEXT:   allocations:
// CHECK-NEXT:     %l0:i32:i 
// CHECK-NEXT:     %l1:i32:j 
// CHECK-NEXT:     %l2:i32:k 
int foo(int i, int j, int k) { 
  // CHECK-LABEL: block b0:
  // CHECK-NEXT:   %b0:p0:i32:i
  // CHECK-NEXT:   %b0:p1:i32:j
  // CHECK-NEXT:   %b0:p2:i32:k 
  // CHECK-NEXT:   %b0:i0:unit = store %b0:p0:i32 %l0:i32* 
  // CHECK-NEXT:   %b0:i1:unit = store %b0:p1:i32 %l1:i32* 
  // CHECK-NEXT:   %b0:i2:unit = store %b0:p2:i32 %l2:i32* 
  // CHECK-NEXT:   %b0:i3:i32 = load %l0:i32* 
  // CHECK-NEXT:   %b0:i4:i32 = load %l1:i32* 
  // CHECK-NEXT:   %b0:i5:i32 = add %b0:i3:i32 %b0:i4:i32
  // CHECK-NEXT:   %b0:i6:i32 = load %l2:i32*
  // CHECK-NEXT:   %b0:i7:i32 = add %b0:i5:i32 %b0:i6:i32
  // CHECK-NEXT:   ret %b0:i7:i32
  return i + j + k; 
}
// CHECK-NEXT: }

// CHECK: fun [ret:i32 params:(i32, i32, i32)]* @foo2 () {
// CHECK-NEXT: init:
// CHECK-NEXT:   bid: b0
// CHECK-NEXT:   allocations:
int (*foo2())(int, int, int) { 
  // CHECK-LABEL: block b0:
  // CHECK-NEXT:   ret @foo:[ret:i32 params:(i32, i32, i32)]*
  return foo; 
}
// CHECK-NEXT: }

// CHECK: fun [ret:[ret:i32 params:(i32, i32, i32)]* params:()]* @foo3 () {
// CHECK-NEXT: init:
// CHECK-NEXT:   bid: b0
// CHECK-NEXT:   allocations:
int (*(*foo3())())(int, int, int) { 
  // CHECK-LABEL: block b0:
  // CHECK-NEXT:   ret @foo2:[ret:[ret:i32 params:(i32, i32, i32)]* params:()]*
  return foo2; 
}
// CHECK-NEXT: }

// CHECK: fun i32 @main () {
// CHECK-NEXT: init:
// CHECK-NEXT:   bid: b0
// CHECK-NEXT:   allocations:
int main() { 
  // CHECK-LABEL: block b0:
  // CHECK-NEXT:   %b0:i0:[ret:[ret:i32 params:(i32, i32, i32)]* params:()]* = call @foo3:[ret:[ret:[ret:i32 params:(i32, i32, i32)]* params:()]* params:()]*()
  // CHECK-NEXT:   %b0:i1:[ret:i32 params:(i32, i32, i32)]* = call %b0:i0:[ret:[ret:i32 params:(i32, i32, i32)]* params:()]*()
  // CHECK-NEXT:   %b0:i2:i32 = call %b0:i1:[ret:i32 params:(i32, i32, i32)]*(2:i32, 2:i32, 2:i32)
  // CHECK-NEXT:   %b0:i3:i1 = cmp eq %b0:i2:i32 6:i32
  // CHECK-NEXT:   %b0:i4:i32 = typecast %b0:i3:i1 to i32
  // CHECK-NEXT:   ret %b0:i4:i32
  return foo3()()(2, 2, 2) == 6; 
}
// CHECK-NEXT: }
