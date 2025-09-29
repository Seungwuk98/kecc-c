// RUN: kecc %s -S -emit-kecc -print-stdout | FileCheck %s
// RUN: keci %s --test-return-value=1
// clang-format off

// CHECK-DAG: struct Sub : { m1:i32, m2:i32, m3:i32, m4:i32 }
// CHECK-DAG: struct Big : { m1:struct Sub, m2:struct Sub, m3:struct Sub }

struct Sub {
  long m1;
  long m2;
  long m3;
  long m4;
};

struct Big {
  struct Sub m1;
  struct Sub m2;
  struct Sub m3;
};

// CHECK: fun struct Big @foo (struct Big) {
// CHECK-NEXT: init:
// CHECK-NEXT:   bid: b0
// CHECK-NEXT:   allocations:
// CHECK-NEXT:    %l0:struct Big:p1
// CHECK-NEXT:    %l1:struct Big:r 
struct Big foo(struct Big p1) {
  // CHECK-LABEL: block b0:
  // CHECK-NEXT:   %b0:p0:struct Big:p1
  // CHECK-NEXT:   %b0:i0:unit = store %b0:p0:struct Big %l0:struct Big*
  struct Big r = p1;
  // CHECK-NEXT:   %b0:i1:struct Big = load %l0:struct Big*
  // CHECK-NEXT:   %b0:i2:unit = store %b0:i1:struct Big %l1:struct Big*

  r.m1.m1 = 10;
  // CHECK-NEXT:   %b0:i3:struct Sub* = getelementptr %l1:struct Big* offset 0:i64
  // CHECK-NEXT:   %b0:i4:i32* = getelementptr %b0:i3:struct Sub* offset 0:i64
  // CHECK-NEXT:   %b0:i5:unit = store 10:i32 %b0:i4:i32*
  return r;
  // CHECK-NEXT:   %b0:i6:struct Big = load %l1:struct Big*
  // CHECK-NEXT:   ret %b0:i6:struct Big
}

// CHECK: fun i32 @main () {
// CHECK-NEXT: init:
// CHECK-NEXT:   bid: b0
// CHECK-NEXT:   allocations:
// CHECK-NEXT:    %l0:struct Big:a
// CHECK-NEXT:    %l1:struct Big:r
int main() {
  // CHECK-LABEL: block b0:
  struct Big a = {{1, 2, 3, 4}, {2, 3, 4, 5}, {3, 4, 5, 6}};
  // CHECK-NEXT:   %b0:i0:struct Sub* = getelementptr %l0:struct Big* offset 0:i64
  // CHECK-NEXT:   %b0:i1:i32* = getelementptr %b0:i0:struct Sub* offset 0:i64
  // CHECK-NEXT:   %b0:i2:unit = store 1:i32 %b0:i1:i32*
  // CHECK-NEXT:   %b0:i3:i32* = getelementptr %b0:i0:struct Sub* offset 4:i64
  // CHECK-NEXT:   %b0:i4:unit = store 2:i32 %b0:i3:i32*
  // CHECK-NEXT:   %b0:i5:i32* = getelementptr %b0:i0:struct Sub* offset 8:i64
  // CHECK-NEXT:   %b0:i6:unit = store 3:i32 %b0:i5:i32*
  // CHECK-NEXT:   %b0:i7:i32* = getelementptr %b0:i0:struct Sub* offset 12:i64
  // CHECK-NEXT:   %b0:i8:unit = store 4:i32 %b0:i7:i32*
  // CHECK-NEXT:   %b0:i9:struct Sub* = getelementptr %l0:struct Big* offset 16:i64
  // CHECK-NEXT:   %b0:i10:i32* = getelementptr %b0:i9:struct Sub* offset 0:i64
  // CHECK-NEXT:   %b0:i11:unit = store 2:i32 %b0:i10:i32*
  // CHECK-NEXT:   %b0:i12:i32* = getelementptr %b0:i9:struct Sub* offset 4:i64
  // CHECK-NEXT:   %b0:i13:unit = store 3:i32 %b0:i12:i32*
  // CHECK-NEXT:   %b0:i14:i32* = getelementptr %b0:i9:struct Sub* offset 8:i64
  // CHECK-NEXT:   %b0:i15:unit = store 4:i32 %b0:i14:i32*
  // CHECK-NEXT:   %b0:i16:i32* = getelementptr %b0:i9:struct Sub* offset 12:i64
  // CHECK-NEXT:   %b0:i17:unit = store 5:i32 %b0:i16:i32*
  // CHECK-NEXT:   %b0:i18:struct Sub* = getelementptr %l0:struct Big* offset 32:i64
  // CHECK-NEXT:   %b0:i19:i32* = getelementptr %b0:i18:struct Sub* offset 0:i64
  // CHECK-NEXT:   %b0:i20:unit = store 3:i32 %b0:i19:i32*
  // CHECK-NEXT:   %b0:i21:i32* = getelementptr %b0:i18:struct Sub* offset 4:i64
  // CHECK-NEXT:   %b0:i22:unit = store 4:i32 %b0:i21:i32*
  // CHECK-NEXT:   %b0:i23:i32* = getelementptr %b0:i18:struct Sub* offset 8:i64
  // CHECK-NEXT:   %b0:i24:unit = store 5:i32 %b0:i23:i32*
  // CHECK-NEXT:   %b0:i25:i32* = getelementptr %b0:i18:struct Sub* offset 12:i64
  // CHECK-NEXT:   %b0:i26:unit = store 6:i32 %b0:i25:i32*
  struct Big r = foo(a);
  // CHECK-NEXT:   %b0:i27:struct Big = load %l0:struct Big*
  // CHECK-NEXT:   %b0:i28:struct Big = call @foo:[ret:struct Big params:(struct Big)]*(%b0:i27:struct Big)
  // CHECK-NEXT:   %b0:i29:unit = store %b0:i28:struct Big %l1:struct Big*
  return r.m1.m1 == 10;
  // CHECK-NEXT:   %b0:i30:struct Sub* = getelementptr %l1:struct Big* offset 0:i64
  // CHECK-NEXT:   %b0:i31:i32* = getelementptr %b0:i30:struct Sub* offset 0:i64
  // CHECK-NEXT:   %b0:i32:i32 = load %b0:i31:i32*
  // CHECK-NEXT:   %b0:i33:i1 = cmp eq %b0:i32:i32 10:i32
  // CHECK-NEXT:   %b0:i34:i32 = typecast %b0:i33:i1 to i32
  // CHECK-NEXT:   ret %b0:i34:i32
}
// CHECK-NEXT: }
