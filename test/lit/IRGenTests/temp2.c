// RUN: kecc %s -S -emit-kecc -print-stdout | FileCheck %s
// RUN: keci %s --test-return-value=7
// clang-format off

struct color {
  int number;
  char name;
};
// CHECK-LABEL: struct color : { number:i32, name:u8 } 

// CHECK: fun i32 @main () {
// CHECK-NEXT: init:
// CHECK-NEXT:   bid: b0
// CHECK-NEXT:   allocations:
// CHECK-NEXT:    %l0:i32:temp
// CHECK-NEXT:    %l1:struct color:c 
// CHECK-NEXT:    %l2:struct color*:cp 
// CHECK-NEXT:    %l3:i32:i
// CHECK-NEXT:    %l4:i32:j
// CHECK-NEXT:    %l5:i1
int main() {
  // CHECK-LABEL: block b0:
  int temp = 0;
  // CHECK-NEXT:   %b0:i0:unit = store 0:i32 %l0:i32*
  temp += sizeof(unsigned char);
  // CHECK-NEXT:   %b0:i1:i32 = load %l0:i32*
  // CHECK-NEXT:   %b0:i2:i32 = add %b0:i1:i32 1:i32
  // CHECK-NEXT:   %b0:i3:unit = store %b0:i2:i32 %l0:i32*
  temp += _Alignof(unsigned char);
  // CHECK-NEXT:   %b0:i4:i32 = load %l0:i32*
  // CHECK-NEXT:   %b0:i5:i32 = add %b0:i4:i32 1:i32
  // CHECK-NEXT:   %b0:i6:unit = store %b0:i5:i32 %l0:i32*

  struct color c = {1, 2};
  // CHECK-NEXT:   %b0:i7:i32* = getelementptr %l1:struct color* offset 0:i64
  // CHECK-NEXT:   %b0:i8:unit = store 1:i32 %b0:i7:i32*
  // CHECK-NEXT:   %b0:i9:u8* = getelementptr %l1:struct color* offset 4:i64
  // CHECK-NEXT:   %b0:i10:unit = store 2:u8 %b0:i9:u8*
  temp += c.name;
  // CHECK-NEXT:   %b0:i11:i32 = load %l0:i32*
  // CHECK-NEXT:   %b0:i12:u8* = getelementptr %l1:struct color* offset 4:i64
  // CHECK-NEXT:   %b0:i13:u8 = load %b0:i12:u8*
  // CHECK-NEXT:   %b0:i14:i32 = typecast %b0:i13:u8 to i32
  // CHECK-NEXT:   %b0:i15:i32 = add %b0:i11:i32 %b0:i14:i32
  // CHECK-NEXT:   %b0:i16:unit = store %b0:i15:i32 %l0:i32*
  struct color *cp = &c;
  // CHECK-NEXT:   %b0:i17:unit = store %l1:struct color* %l2:struct color**
  temp += cp->name;
  // CHECK-NEXT:   %b0:i18:i32 = load %l0:i32*
  // CHECK-NEXT:   %b0:i19:struct color* = load %l2:struct color**
  // CHECK-NEXT:   %b0:i20:u8* = getelementptr %b0:i19:struct color* offset 4:i64
  // CHECK-NEXT:   %b0:i21:u8 = load %b0:i20:u8*
  // CHECK-NEXT:   %b0:i22:i32 = typecast %b0:i21:u8 to i32
  // CHECK-NEXT:   %b0:i23:i32 = add %b0:i18:i32 %b0:i22:i32
  // CHECK-NEXT:   %b0:i24:unit = store %b0:i23:i32 %l0:i32*

  // CHECK-NEXT:   %b0:i25:unit = store 0:i32 %l3:i32*
  // CHECK-NEXT:   %b0:i26:unit = store 0:i32 %l4:i32*
  // CHECK-NEXT:   j b1()
  for (int i = 0, j = 0; i < 10; ++i) {
    // CHECK-LABEL: block b1:
    // CHECK-NEXT:   %b1:i0:i32 = load %l3:i32*
    // CHECK-NEXT:   %b1:i1:i1 = cmp lt %b1:i0:i32 10:i32
    // CHECK-NEXT:   %b1:i2:i32 = typecast %b1:i1:i1 to i32
    // CHECK-NEXT:   %b1:i3:i1 = cmp ne %b1:i2:i32 0:i32
    // CHECK-NEXT:   br %b1:i3:i1, b2(), b4()

    // CHECK-LABEL: block b2:
    // CHECK-NEXT:   %b2:i0:i32 = load %l3:i32*
    // CHECK-NEXT:   %b2:i1:i1 = cmp eq %b2:i0:i32 2:i32
    // CHECK-NEXT:   %b2:i2:i32 = typecast %b2:i1:i1 to i32
    // CHECK-NEXT:   %b2:i3:i1 = cmp ne %b2:i2:i32 0:i32
    // CHECK-NEXT:   br %b2:i3:i1, b5(), b6()

    // CHECK-LABEL: block b3:
    // CHECK-NEXT:   %b3:i0:i32 = load %l3:i32*
    // CHECK-NEXT:   %b3:i1:i32 = add %b3:i0:i32 1:i32
    // CHECK-NEXT:   %b3:i2:unit = store %b3:i1:i32 %l3:i32*
    // CHECK-NEXT:   j b1()

    // CHECK-LABEL: block b4:
    // CHECK-NEXT:   %b4:i0:i32 = load %l0:i32*
    // CHECK-NEXT:   switch %b4:i0:i32 default b11() [
    // CHECK-NEXT:     1:i32 b13()
    // CHECK-NEXT:   ]

    // CHECK-LABEL: block b5:
    // CHECK-NEXT:   %b5:i0:i32 = load %l4:i32*
    // CHECK-NEXT:   %b5:i1:i1 = cmp eq %b5:i0:i32 0:i32
    // CHECK-NEXT:   %b5:i2:i32 = typecast %b5:i1:i1 to i32
    // CHECK-NEXT:   %b5:i3:i1 = cmp ne %b5:i2:i32 0:i32
    // CHECK-NEXT:   %b5:i4:unit = store %b5:i3:i1  %l5:i1*
    // CHECK-NEXT:   j b7()

    // CHECK-LABEL: block b6:
    // CHECK-NEXT:   %b6:i0:unit = store 0:i1 %l5:i1*
    // CHECK-NEXT:   j b7()

    // CHECK-LABEL: block b7:
    // CHECK-NEXT:   %b7:i0:i1 = load %l5:i1*
    // CHECK-NEXT:   br %b7:i0:i1, b8(), b9()

    // CHECK-LABEL: block b8:
    // CHECK-NEXT:   j b4()
    if (i == 2 && j == 0)
      break;

    // CHECK-LABEL: block b9:
    // CHECK-NEXT:   j b10()

    // CHECK-LABEL: block b10:
    // CHECK-NEXT:   %b10:i0:i32 = load %l0:i32*
    // CHECK-NEXT:   %b10:i1:i32 = load %l3:i32*
    // CHECK-NEXT:   %b10:i2:i32 = add %b10:i0:i32 %b10:i1:i32
    // CHECK-NEXT:   %b10:i3:unit = store %b10:i2:i32 %l0:i32*
    // CHECK-NEXT:   j b3()
    temp += i;
  }

  // CHECK-LABEL: block b11:
  // CHECK-NEXT:   j b12()
  
  // CHECK-LABEL: block b12:
  // CHECK-NEXT:   %b12:i0:i32 = load %l0:i32*
  // CHECK-NEXT:   ret %b12:i0:i32

  // CHECK-LABEL: block b13:
  // CHECK-NEXT:   %b13:i0:unit = store 0:i32 %l0:i32*
  // CHECK-NEXT:   j b12()
  switch (temp) {
  case 1: {
    temp = 0;
    break;
  }
  default: {
    break;
  }
  }

  return temp;
}
