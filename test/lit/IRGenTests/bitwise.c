// RUN: kecc %s -S -emit-kecc -print-stdout | FileCheck %s
// clang-format off

// CHECK: fun i32 @main () {
// CHECK-NEXT: init:
// CHECK-NEXT:   bid: b0
// CHECK-NEXT:   allocations:
// CHECK-NEXT:     %l0:u8:a
// CHECK-NEXT:     %l1:u8:b 
// CHECK-NEXT:     %l2:u8:c 
// CHECK-NEXT:     %l3:u8:d 
// CHECK-NEXT:     %l4:u8:e 
// CHECK-NEXT:     %l5:u8:f 
// CHECK-NEXT:     %l6:u8:g 
// CHECK-NEXT:     %l7:u8:h 
// CHECK-NEXT:     %l8:u8:i 
// CHECK-NEXT:     %l9:i1
// CHECK-NEXT:     %l10:i1
// CHECK-NEXT:     %l11:i1
// CHECK-NEXT:     %l12:i1
// CHECK-NEXT:     %l13:i1

int main() {
  // CHECK-LABEL: block b0:
  unsigned char a = -1;
  // CHECK-NEXT:   %b0:i0:i32 = minus 1:i32
  // CHECK-NEXT:   %b0:i1:u8 = typecast %b0:i0:i32 to u8
  // CHECK-NEXT:   %b0:i2:unit = store %b0:i1:u8 %l0:u8*
  
  unsigned char b = -128;
  // CHECK-NEXT:   %b0:i3:i32 = minus 128:i32
  // CHECK-NEXT:   %b0:i4:u8 = typecast %b0:i3:i32 to u8
  // CHECK-NEXT:   %b0:i5:unit = store %b0:i4:u8 %l1:u8*
  
  unsigned char c = 127;
  // CHECK-NEXT:   %b0:i6:unit = store 127:u8 %l2:u8*
  
  unsigned char d = b | a;   // -1 (255)
  // CHECK-NEXT:   %b0:i7:u8 = load %l1:u8*
  // CHECK-NEXT:   %b0:i8:i32 = typecast %b0:i7:u8 to i32
  // CHECK-NEXT:   %b0:i9:u8 = load %l0:u8*
  // CHECK-NEXT:   %b0:i10:i32 = typecast %b0:i9:u8 to i32
  // CHECK-NEXT:   %b0:i11:i32 = or %b0:i8:i32 %b0:i10:i32
  // CHECK-NEXT:   %b0:i12:u8 = typecast %b0:i11:i32 to u8
  // CHECK-NEXT:   %b0:i13:unit = store %b0:i12:u8 %l3:u8*

  unsigned char e = b & a;   // -128 (128)
  // CHECK-NEXT:   %b0:i14:u8 = load %l1:u8*
  // CHECK-NEXT:   %b0:i15:i32 = typecast %b0:i14:u8 to i32
  // CHECK-NEXT:   %b0:i16:u8 = load %l0:u8*
  // CHECK-NEXT:   %b0:i17:i32 = typecast %b0:i16:u8 to i32
  // CHECK-NEXT:   %b0:i18:i32 = and %b0:i15:i32 %b0:i17:i32
  // CHECK-NEXT:   %b0:i19:u8 = typecast %b0:i18:i32 to u8
  // CHECK-NEXT:   %b0:i20:unit = store %b0:i19:u8 %l4:u8*

  unsigned char f = b & c;   // 0 (0)
  // CHECK-NEXT:   %b0:i21:u8 = load %l1:u8*
  // CHECK-NEXT:   %b0:i22:i32 = typecast %b0:i21:u8 to i32
  // CHECK-NEXT:   %b0:i23:u8 = load %l2:u8*
  // CHECK-NEXT:   %b0:i24:i32 = typecast %b0:i23:u8 to i32
  // CHECK-NEXT:   %b0:i25:i32 = and %b0:i22:i32 %b0:i24:i32
  // CHECK-NEXT:   %b0:i26:u8 = typecast %b0:i25:i32 to u8
  // CHECK-NEXT:   %b0:i27:unit = store %b0:i26:u8 %l5:u8*

  unsigned char g = b | c;   // -1 (255)
  // CHECK-NEXT:   %b0:i28:u8 = load %l1:u8*
  // CHECK-NEXT:   %b0:i29:i32 = typecast %b0:i28:u8 to i32
  // CHECK-NEXT:   %b0:i30:u8 = load %l2:u8*
  // CHECK-NEXT:   %b0:i31:i32 = typecast %b0:i30:u8 to i32
  // CHECK-NEXT:   %b0:i32:i32 = or %b0:i29:i32 %b0:i31:i32
  // CHECK-NEXT:   %b0:i33:u8 = typecast %b0:i32:i32 to u8
  // CHECK-NEXT:   %b0:i34:unit = store %b0:i33:u8 %l6:u8*

  unsigned char h = -1 ^ -1; // 0 (0)
  // CHECK-NEXT:   %b0:i35:i32 = minus 1:i32
  // CHECK-NEXT:   %b0:i36:i32 = minus 1:i32
  // CHECK-NEXT:   %b0:i37:i32 = xor %b0:i35:i32 %b0:i36:i32
  // CHECK-NEXT:   %b0:i38:u8 = typecast %b0:i37:i32 to u8
  // CHECK-NEXT:   %b0:i39:unit = store %b0:i38:u8 %l7:u8*

  unsigned char i = -1 ^ 0;  // -1 (255)
  // CHECK-NEXT:   %b0:i40:i32 = minus 1:i32
  // CHECK-NEXT:   %b0:i41:i32 = xor %b0:i40:i32 0:i32
  // CHECK-NEXT:   %b0:i42:u8 = typecast %b0:i41:i32 to u8
  // CHECK-NEXT:   %b0:i43:unit = store %b0:i42:u8 %l8:u8*

  return d == 255 && e == 128 && f == 0 && g == 255 && h == 0 && i == 255;
  // CHECK-NEXT:   %b0:i44:u8 = load %l3:u8*
  // CHECK-NEXT:   %b0:i45:i32 = typecast %b0:i44:u8 to i32
  // CHECK-NEXT:   %b0:i46:i1 = cmp eq %b0:i45:i32 255:i32
  // CHECK-NEXT:   %b0:i47:i32 = typecast %b0:i46:i1 to i32
  // CHECK-NEXT:   %b0:i48:i1 = cmp ne %b0:i47:i32 0:i32
  // CHECK-NEXT:   br %b0:i48:i1, b1(), b2()

  // CHECK-LABEL: block b1:
  // CHECK-NEXT:   %b1:i0:u8 = load %l4:u8*
  // CHECK-NEXT:   %b1:i1:i32 = typecast %b1:i0:u8 to i32
  // CHECK-NEXT:   %b1:i2:i1 = cmp eq %b1:i1:i32 128:i32
  // CHECK-NEXT:   %b1:i3:i32 = typecast %b1:i2:i1 to i32
  // CHECK-NEXT:   %b1:i4:i1 = cmp ne %b1:i3:i32 0:i32
  // CHECK-NEXT:   %b1:i5:unit = store %b1:i4:i1 %l9:i1*
  // CHECK-NEXT:   j b3()

  // CHECK-LABEL: block b2:
  // CHECK-NEXT:   %b2:i0:unit = store 0:i1 %l9:i1*
  // CHECK-NEXT:   j b3()

  // CHECK-LABEL: block b3:
  // CHECK-NEXT:   %b3:i0:i1 = load %l9:i1*
  // CHECK-NEXT:   br %b3:i0:i1, b4(), b5()
  
  // CHECK-LABEL: block b4: 
  // CHECK-NEXT:   %b4:i0:u8 = load %l5:u8*
  // CHECK-NEXT:   %b4:i1:i32 = typecast %b4:i0:u8 to i32
  // CHECK-NEXT:   %b4:i2:i1 = cmp eq %b4:i1:i32 0:i32
  // CHECK-NEXT:   %b4:i3:i32 = typecast %b4:i2:i1 to i32
  // CHECK-NEXT:   %b4:i4:i1 = cmp ne %b4:i3:i32 0:i32
  // CHECK-NEXT:   %b4:i5:unit = store %b4:i4:i1 %l10:i1*
  // CHECK-NEXT:   j b6()

  // CHECK-LABEL: block b5:
  // CHECK-NEXT:   %b5:i0:unit = store 0:i1 %l10:i1* 
  // CHECK-NEXT:   j b6()

  // CHECK-LABEL: block b6:
  // CHECK-NEXT:   %b6:i0:i1 = load %l10:i1*
  // CHECK-NEXT:   br %b6:i0:i1, b7(), b8()

  // CHECK-LABEL: block b7: 
  // CHECK-NEXT:   %b7:i0:u8 = load %l6:u8*
  // CHECK-NEXT:   %b7:i1:i32 = typecast %b7:i0:u8 to i32
  // CHECK-NEXT:   %b7:i2:i1 = cmp eq %b7:i1:i32 255:i32
  // CHECK-NEXT:   %b7:i3:i32 = typecast %b7:i2:i1 to i32
  // CHECK-NEXT:   %b7:i4:i1 = cmp ne %b7:i3:i32 0:i32
  // CHECK-NEXT:   %b7:i5:unit = store %b7:i4:i1 %l11:i1*
  // CHECK-NEXT:   j b9()
  
  // CHECK-LABEL: block b8:
  // CHECK-NEXT:   %b8:i0:unit = store 0:i1 %l11:i1*
  // CHECK-NEXT:   j b9()

  // CHECK-LABEL: block b9:
  // CHECK-NEXT:   %b9:i0:i1 = load %l11:i1*
  // CHECK-NEXT:   br %b9:i0:i1, b10(), b11()

  // CHECK-LABEL: block b10:
  // CHECK-NEXT:   %b10:i0:u8 = load %l7:u8*
  // CHECK-NEXT:   %b10:i1:i32 = typecast %b10:i0:u8 to i32
  // CHECK-NEXT:   %b10:i2:i1 = cmp eq %b10:i1:i32 0:i32
  // CHECK-NEXT:   %b10:i3:i32 = typecast %b10:i2:i1 to i32
  // CHECK-NEXT:   %b10:i4:i1 = cmp ne %b10:i3:i32 0:i32
  // CHECK-NEXT:   %b10:i5:unit = store %b10:i4:i1 %l12:i1*
  // CHECK-NEXT:   j b12()

  // CHECK-LABEL: block b11:
  // CHECK-NEXT:   %b11:i0:unit = store 0:i1 %l12:i1*
  // CHECK-NEXT:   j b12()
  
  // CHECK-LABEL: block b12:
  // CHECK-NEXT:   %b12:i0:i1 = load %l12:i1*
  // CHECK-NEXT:   br %b12:i0:i1, b13(), b14()
  
  // CHECK-LABEL: block b13: 
  // CHECK-NEXT:   %b13:i0:u8 = load %l8:u8*
  // CHECK-NEXT:   %b13:i1:i32 = typecast %b13:i0:u8 to i32
  // CHECK-NEXT:   %b13:i2:i1 = cmp eq %b13:i1:i32 255:i32
  // CHECK-NEXT:   %b13:i3:i32 = typecast %b13:i2:i1 to i32
  // CHECK-NEXT:   %b13:i4:i1 = cmp ne %b13:i3:i32 0:i32
  // CHECK-NEXT:   %b13:i5:unit = store %b13:i4:i1 %l13:i1*
  // CHECK-NEXT:   j b15()

  // CHECK-LABEL: block b14:
  // CHECK-NEXT:   %b14:i0:unit = store 0:i1 %l13:i1* 
  // CHECK-NEXT:   j b15()

  // CHECK-LABEL: block b15:
  // CHECK-NEXT:   %b15:i0:i1 = load %l13:i1*
  // CHECK-NEXT:   %b15:i1:i32 = typecast %b15:i0:i1 to i32
  // CHECK-NEXT:   ret %b15:i1:i32
 }
