// RUN: kecc %s -S -emit-kecc -print-stdout | FileCheck %s
// RUN: keci %s --test-return-value=1
// clang-format off

// CHECK-DAG: struct %t0 : { b:[4 x i32] }
// CHECK-DAG: struct %t1 : { a:u8, %anon:struct %t0, c:i32 }
typedef struct {
  char a;
  struct {
    int b[4];
  };
  long c;
} Temp;

// CHECK: fun i32 @main () {
// CHECK-NEXT: init:
// CHECK-NEXT:   bid: b0
// CHECK-NEXT:   allocations:
// CHECK-NEXT:    %l0:const struct %t1:temp
// CHECK-NEXT:    %l1:struct %t1:temp2
// CHECK-NEXT:    %l2:i32:sum
int main() {
  // CHECK-LABEL: block b0:
  // CHECK-NEXT:   %b0:i0:u8* = getelementptr %l0:const struct %t1* offset 0:i64
  // CHECK-NEXT:   %b0:i1:unit = store 1:u8 %b0:i0:u8*
  // CHECK-NEXT:   %b0:i2:struct %t0* = getelementptr %l0:const struct %t1* offset 4:i64
  // CHECK-NEXT:   %b0:i3:[4 x i32]* = getelementptr %b0:i2:struct %t0* offset 0:i64
  // CHECK-NEXT:   %b0:i4:i32* = getelementptr %b0:i3:[4 x i32]* offset 0:i64
  // CHECK-NEXT:   %b0:i5:unit = store 2:i32 %b0:i4:i32*
  // CHECK-NEXT:   %b0:i6:i32* = getelementptr %b0:i3:[4 x i32]* offset 4:i64
  // CHECK-NEXT:   %b0:i7:unit = store 3:i32 %b0:i6:i32*
  // CHECK-NEXT:   %b0:i8:i32* = getelementptr %b0:i3:[4 x i32]* offset 8:i64
  // CHECK-NEXT:   %b0:i9:unit = store 4:i32 %b0:i8:i32*
  // CHECK-NEXT:   %b0:i10:i32* = getelementptr %b0:i3:[4 x i32]* offset 12:i64
  // CHECK-NEXT:   %b0:i11:unit = store 5:i32 %b0:i10:i32*
  // CHECK-NEXT:   %b0:i12:i32* = getelementptr %l0:const struct %t1* offset 20:i64
  // CHECK-NEXT:   %b0:i13:unit = store 6:i32 %b0:i12:i32*
  const Temp temp = {1, {{2, 3, 4, 5}}, 6};

  Temp temp2;
  temp2 = temp;
  // CHECK-NEXT:   %b0:i14:struct %t1 = load %l0:const struct %t1*
  // CHECK-NEXT:   %b0:i15:unit = store %b0:i14:struct %t1 %l1:struct %t1*

  int sum = temp2.a + temp2.b[2] + temp2.c;
  // CHECK-NEXT:   %b0:i16:u8* = getelementptr %l1:struct %t1* offset 0:i64
  // CHECK-NEXT:   %b0:i17:u8 = load %b0:i16:u8*
  // CHECK-NEXT:   %b0:i18:i32 = typecast %b0:i17:u8 to i32
  // CHECK-NEXT:   %b0:i19:struct %t0* = getelementptr %l1:struct %t1* offset 4:i64
  // CHECK-NEXT:   %b0:i20:[4 x i32]* = getelementptr %b0:i19:struct %t0* offset 0:i64
  // CHECK-NEXT:   %b0:i21:i32* = getelementptr %b0:i20:[4 x i32]* offset 0:i64
  // CHECK-NEXT:   %b0:i22:i64 = typecast 2:i32 to i64
  // CHECK-NEXT:   %b0:i23:i64 = mul %b0:i22:i64 4:i64
  // CHECK-NEXT:   %b0:i24:i32* = getelementptr %b0:i21:i32* offset %b0:i23:i64
  // CHECK-NEXT:   %b0:i25:i32 = load %b0:i24:i32*
  // CHECK-NEXT:   %b0:i26:i32 = add %b0:i18:i32 %b0:i25:i32
  // CHECK-NEXT:   %b0:i27:i32* = getelementptr %l1:struct %t1* offset 20:i64
  // CHECK-NEXT:   %b0:i28:i32 = load %b0:i27:i32*
  // CHECK-NEXT:   %b0:i29:i32 = add %b0:i26:i32 %b0:i28:i32
  // CHECK-NEXT:   %b0:i30:unit = store %b0:i29:i32 %l2:i32*

  // CHECK-NEXT:   %b0:i31:i32 = load %l2:i32*
  // CHECK-NEXT:   %b0:i32:i1 = cmp eq %b0:i31:i32 11:i32
  // CHECK-NEXT:   %b0:i33:i32 = typecast %b0:i32:i1 to i32
  // CHECK-NEXT:   ret %b0:i33:i32
  return sum == 11;
}
// CHECK-NEXT: }
