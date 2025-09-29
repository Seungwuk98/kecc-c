// RUN: kecc %s -S -emit-kecc -print-stdout | FileCheck %s
// RUN: keci %s --test-return-value=1
// clang-format off

// CHECK-DAG: struct %t0 : { b:[4 x [5 x i32]] }
// CHECK-DAG: struct %t1 : { a:u8, %anon:struct %t0, c:f64 }
typedef struct {
  char a;
  struct {
    int b[4][5];
  };
  double c;
} Temp;

// CHECK: fun unit @init (i32, i32, [5 x i32]*) {
// CHECK-NEXT: init:
// CHECK-NEXT:   bid: b0
// CHECK-NEXT:   allocations:
// CHECK-NEXT:    %l0:i32:row
// CHECK-NEXT:    %l1:i32:col
// CHECK-NEXT:    %l2:[5 x i32]*:arr
// CHECK-NEXT:    %l3:i32:i
// CHECK-NEXT:    %l4:i32:j
void init(int row, int col, int arr[4][5]) {
  // CHECK-LABEL: block b0:
  // CHECK-NEXT:   %b0:p0:i32:row
  // CHECK-NEXT:   %b0:p1:i32:col
  // CHECK-NEXT:   %b0:p2:[5 x i32]*:arr
  // CHECK-NEXT:   %b0:i0:unit = store %b0:p0:i32 %l0:i32*
  // CHECK-NEXT:   %b0:i1:unit = store %b0:p1:i32 %l1:i32*
  // CHECK-NEXT:   %b0:i2:unit = store %b0:p2:[5 x i32]* %l2:[5 x i32]**

  // CHECK-NEXT:   %b0:i3:unit = store 0:i32 %l3:i32*
  // CHECK-NEXT:   j b1()
  for (int i = 0; i < row; i++) {
    // CHECK-LABEL: block b1:
    // CHECK-NEXT:   %b1:i0:i32 = load %l3:i32*
    // CHECK-NEXT:   %b1:i1:i32 = load %l0:i32*
    // CHECK-NEXT:   %b1:i2:i1 = cmp lt %b1:i0:i32 %b1:i1:i32
    // CHECK-NEXT:   %b1:i3:i32 = typecast %b1:i2:i1 to i32
    // CHECK-NEXT:   %b1:i4:i1 = cmp ne %b1:i3:i32 0:i32
    // CHECK-NEXT:   br %b1:i4:i1, b2(), b4()

    // CHECK-LABEL: block b2:
    // CHECK-NEXT:   %b2:i0:unit = store 0:i32 %l4:i32*
    // CHECK-NEXT:   j b5()

    // CHECK-LABEL: block b3:
    // CHECK-NEXT:   %b3:i0:i32 = load %l3:i32*
    // CHECK-NEXT:   %b3:i1:i32 = add %b3:i0:i32 1:i32
    // CHECK-NEXT:   %b3:i2:unit = store %b3:i1:i32 %l3:i32*
    // CHECK-NEXT:   j b1()

    // CHECK-LABEL: block b4:
    // CHECK-NEXT:   ret unit:unit
    for (int j = 0; j < col; j++) {
      // CHECK-LABEL: block b5:
      // CHECK-NEXT:   %b5:i0:i32 = load %l4:i32*
      // CHECK-NEXT:   %b5:i1:i32 = load %l1:i32*
      // CHECK-NEXT:   %b5:i2:i1 = cmp lt %b5:i0:i32 %b5:i1:i32
      // CHECK-NEXT:   %b5:i3:i32 = typecast %b5:i2:i1 to i32
      // CHECK-NEXT:   %b5:i4:i1 = cmp ne %b5:i3:i32 0:i32
      // CHECK-NEXT:   br %b5:i4:i1, b6(), b8()

      // CHECK-LABEL: block b6:
      // CHECK-NEXT:   %b6:i0:[5 x i32]* = load %l2:[5 x i32]**
      // CHECK-NEXT:   %b6:i1:i32 = load %l3:i32*
      // CHECK-NEXT:   %b6:i2:i64 = typecast %b6:i1:i32 to i64
      // CHECK-NEXT:   %b6:i3:i64 = mul %b6:i2:i64 20:i64
      // CHECK-NEXT:   %b6:i4:[5 x i32]* = getelementptr %b6:i0:[5 x i32]* offset %b6:i3:i64
      // CHECK-NEXT:   %b6:i5:i32* = getelementptr %b6:i4:[5 x i32]* offset 0:i64
      // CHECK-NEXT:   %b6:i6:i32 = load %l4:i32*
      // CHECK-NEXT:   %b6:i7:i64 = typecast %b6:i6:i32 to i64
      // CHECK-NEXT:   %b6:i8:i64 = mul %b6:i7:i64 4:i64
      // CHECK-NEXT:   %b6:i9:i32* = getelementptr %b6:i5:i32* offset %b6:i8:i64
      // CHECK-NEXT:   %b6:i10:i32 = load %l3:i32*
      // CHECK-NEXT:   %b6:i11:i32 = load %l4:i32*
      // CHECK-NEXT:   %b6:i12:i32 = mul %b6:i10:i32 %b6:i11:i32
      // CHECK-NEXT:   %b6:i13:unit = store %b6:i12:i32 %b6:i9:i32*
      // CHECK-NEXT:   j b7()
      arr[i][j] = i * j;

      // CHECK-LABEL: block b7:
      // CHECK-NEXT:   %b7:i0:i32 = load %l4:i32*
      // CHECK-NEXT:   %b7:i1:i32 = add %b7:i0:i32 1:i32
      // CHECK-NEXT:   %b7:i2:unit = store %b7:i1:i32 %l4:i32*
      // CHECK-NEXT:   j b5()
    }
  }
}

// CHECK: fun i32 @main () {
// CHECK-NEXT: init:
// CHECK-NEXT:   bid: b0
// CHECK-NEXT:   allocations:
// CHECK-NEXT:    %l0:struct %t1:temp
// CHECK-NEXT:    %l1:i32:row 
// CHECK-NEXT:    %l2:i32:col 
// CHECK-NEXT:    %l3:struct %t1:temp2
int main() {
  // CHECK-LABEL: block b0:
  Temp temp;
  int row = 4, col = 5;
  // CHECK-NEXT:   %b0:i0:unit = store 4:i32 %l1:i32*
  // CHECK-NEXT:   %b0:i1:unit = store 5:i32 %l2:i32*
  init(row, col, temp.b);
  // CHECK-NEXT:   %b0:i2:i32 = load %l1:i32*
  // CHECK-NEXT:   %b0:i3:i32 = load %l2:i32*
  // CHECK-NEXT:   %b0:i4:struct %t0* = getelementptr %l0:struct %t1* offset 4:i64
  // CHECK-NEXT:   %b0:i5:[4 x [5 x i32]]* = getelementptr %b0:i4:struct %t0* offset 0:i64
  // CHECK-NEXT:   %b0:i6:[5 x i32]* = getelementptr %b0:i5:[4 x [5 x i32]]* offset 0:i64
  // CHECK-NEXT:   %b0:i7:unit = call @init:[ret:unit params:(i32, i32, [5 x i32]*)]*(%b0:i2:i32, %b0:i3:i32, %b0:i6:[5 x i32]*)

  Temp temp2;
  temp2 = temp;
  // CHECK-NEXT:   %b0:i8:struct %t1 = load %l0:struct %t1*
  // CHECK-NEXT:   %b0:i9:unit = store %b0:i8:struct %t1 %l3:struct %t1*

  // CHECK-NEXT:   %b0:i10:struct %t0* = getelementptr %l3:struct %t1* offset 4:i64
  // CHECK-NEXT:   %b0:i11:[4 x [5 x i32]]* = getelementptr %b0:i10:struct %t0* offset 0:i64
  // CHECK-NEXT:   %b0:i12:[5 x i32]* = getelementptr %b0:i11:[4 x [5 x i32]]* offset 0:i64
  // CHECK-NEXT:   %b0:i13:i64 = typecast 2:i32 to i64
  // CHECK-NEXT:   %b0:i14:i64 = mul %b0:i13:i64 20:i64
  // CHECK-NEXT:   %b0:i15:[5 x i32]* = getelementptr %b0:i12:[5 x i32]* offset %b0:i14:i64
  // CHECK-NEXT:   %b0:i16:i32* = getelementptr %b0:i15:[5 x i32]* offset 0:i64
  // CHECK-NEXT:   %b0:i17:i64 = typecast 3:i32 to i64
  // CHECK-NEXT:   %b0:i18:i64 = mul %b0:i17:i64 4:i64
  // CHECK-NEXT:   %b0:i19:i32* = getelementptr %b0:i16:i32* offset %b0:i18:i64
  // CHECK-NEXT:   %b0:i20:i32 = load %b0:i19:i32*
  // CHECK-NEXT:   %b0:i21:i1 = cmp eq %b0:i20:i32 6:i32
  // CHECK-NEXT:   %b0:i22:i32 = typecast %b0:i21:i1 to i32
  // CHECK-NEXT:   ret %b0:i22:i32
  return temp2.b[2][3] == 6;
}
