// RUN: kecc %s -S -emit-kecc -print-stdout | FileCheck %s
// clang-format off

// CHECK: fun f64 @custom_abs (f64) {
// CHECK-NEXT: init:
// CHECK-NEXT:   bid: b0
// CHECK-NEXT:   allocations:
// CHECK-NEXT:     %l0:f64:a
// CHECK-NEXT:     %l1:f64
double custom_abs(double a) { 
  // CHECK-LABEL: block b0:
  // CHECK-NEXT:   %b0:p0:f64:a 
  // CHECK-NEXT:   %b0:i0:unit = store %b0:p0:f64 %l0:f64* 
  // CHECK-NEXT:   %b0:i1:f64 = load %l0:f64*
  // CHECK-NEXT:   %b0:i2:i1 = cmp lt %b0:i1:f64 0:f64
  // CHECK-NEXT:   %b0:i3:i32 = typecast %b0:i2:i1 to i32 
  // CHECK-NEXT:   %b0:i4:i1 = cmp ne %b0:i3:i32 0:i32
  // CHECK-NEXT:   br %b0:i4:i1, b1(), b2()

  // CHECK-LABEL: block b1:
  // CHECK-NEXT:   %b1:i0:f64 = load %l0:f64*
  // CHECK-NEXT:   %b1:i1:f64 = minus %b1:i0:f64
  // CHECK-NEXT:   %b1:i2:unit = store %b1:i1:f64 %l1:f64*
  // CHECK-NEXT:   j b3()

  // CHECK-LABEL: block b2:
  // CHECK-NEXT:   %b2:i0:f64 = load %l0:f64*
  // CHECK-NEXT:   %b2:i1:unit = store %b2:i0:f64 %l1:f64*
  // CHECK-NEXT:   j b3()

  // CHECK-LABEL: block b3:
  // CHECK-NEXT:   %b3:i0:f64 = load %l1:f64*
  // CHECK-NEXT:   ret %b3:i0:f64
  return a < 0 ? -a : a; 
}
// CHECK-NEXT: }

// CHECK: fun f64 @custom_max (f64, f64) {
// CHECK-NEXT: init:
// CHECK-NEXT:   bid: b0
// CHECK-NEXT:   allocations:
// CHECK-NEXT:     %l0:f64:a
// CHECK-NEXT:     %l1:f64:b 
// CHECK-NEXT:     %l2:f64
double custom_max(double a, double b) { 
  // CHECK-LABEL: block b0:
  // CHECK-NEXT:   %b0:p0:f64:a
  // CHECK-NEXT:   %b0:p1:f64:b 
  // CHECK-NEXT:   %b0:i0:unit = store %b0:p0:f64 %l0:f64* 
  // CHECK-NEXT:   %b0:i1:unit = store %b0:p1:f64 %l1:f64*
  // CHECK-NEXT:   %b0:i2:f64 = load %l0:f64*
  // CHECK-NEXT:   %b0:i3:f64 = load %l1:f64*
  // CHECK-NEXT:   %b0:i4:i1 = cmp gt %b0:i2:f64 %b0:i3:f64
  // CHECK-NEXT:   %b0:i5:i32 = typecast %b0:i4:i1 to i32
  // CHECK-NEXT:   %b0:i6:i1 = cmp ne %b0:i5:i32 0:i32
  // CHECK-NEXT:   br %b0:i6:i1, b1(), b2()

  // CHECK-LABEL: block b1:
  // CHECK-NEXT:   %b1:i0:f64 = load %l0:f64*
  // CHECK-NEXT:   %b1:i1:unit = store %b1:i0:f64 %l2:f64*
  // CHECK-NEXT:   j b3()
  
  // CHECK-LABEL: block b2:
  // CHECK-NEXT:   %b2:i0:f64 = load %l1:f64*
  // CHECK-NEXT:   %b2:i1:unit = store %b2:i0:f64 %l2:f64*
  // CHECK-NEXT:   j b3()

  // CHECK-LABEL: block b3:
  // CHECK-NEXT:   %b3:i0:f64 = load %l2:f64*
  // CHECK-NEXT:   ret %b3:i0:f64
  return a > b ? a : b; 
}
// CHECK-NEXT: }

// CHECK: fun i32 @is_close (f64, f64, f64, f64) {
// CHECK-NEXT: init:
// CHECK-NEXT:   bid: b0
// CHECK-NEXT:   allocations:
// CHECK-NEXT:     %l0:f64:a
// CHECK-NEXT:     %l1:f64:b 
// CHECK-NEXT:     %l2:f64:rel_tol 
// CHECK-NEXT:     %l3:f64:abs_tol
int is_close(double a, double b, double rel_tol, double abs_tol) {
  // CHECK-LABEL: block b0:
  // CHECK-NEXT:   %b0:p0:f64:a 
  // CHECK-NEXT:   %b0:p1:f64:b 
  // CHECK-NEXT:   %b0:p2:f64:rel_tol 
  // CHECK-NEXT:   %b0:p3:f64:abs_tol
  // CHECK-NEXT:   %b0:i0:unit = store %b0:p0:f64 %l0:f64*
  // CHECK-NEXT:   %b0:i1:unit = store %b0:p1:f64 %l1:f64*
  // CHECK-NEXT:   %b0:i2:unit = store %b0:p2:f64 %l2:f64*
  // CHECK-NEXT:   %b0:i3:unit = store %b0:p3:f64 %l3:f64*
  // CHECK-NEXT:   %b0:i4:f64 = load %l0:f64*
  // CHECK-NEXT:   %b0:i5:f64 = load %l1:f64*
  // CHECK-NEXT:   %b0:i6:f64 = sub %b0:i4:f64 %b0:i5:f64
  // CHECK-NEXT:   %b0:i7:f64 = call @custom_abs:[ret:f64 params:(f64)]*(%b0:i6:f64)
  // CHECK-NEXT:   %b0:i8:f64 = load %l2:f64*
  // CHECK-NEXT:   %b0:i9:f64 = load %l0:f64*
  // CHECK-NEXT:   %b0:i10:f64 = call @custom_abs:[ret:f64 params:(f64)]*(%b0:i9:f64)
  // CHECK-NEXT:   %b0:i11:f64 = load %l1:f64*
  // CHECK-NEXT:   %b0:i12:f64 = call @custom_abs:[ret:f64 params:(f64)]*(%b0:i11:f64)
  // CHECK-NEXT:   %b0:i13:f64 = call @custom_max:[ret:f64 params:(f64, f64)]*(%b0:i10:f64, %b0:i12:f64)
  // CHECK-NEXT:   %b0:i14:f64 = mul %b0:i8:f64 %b0:i13:f64
  // CHECK-NEXT:   %b0:i15:f64 = load %l3:f64*
  // CHECK-NEXT:   %b0:i16:f64 = call @custom_max:[ret:f64 params:(f64, f64)]*(%b0:i14:f64, %b0:i15:f64)
  // CHECK-NEXT:   %b0:i17:i1 = cmp le %b0:i7:f64 %b0:i16:f64
  // CHECK-NEXT:   %b0:i18:i32 = typecast %b0:i17:i1 to i32
  // CHECK-NEXT:   ret %b0:i18:i32
  return custom_abs(a - b) <=
         custom_max(rel_tol * custom_max(custom_abs(a), custom_abs(b)),
                    abs_tol);
}
// CHECK-NEXT: }

// CHECK: fun f64 @average (i32, i32*) {
// CHECK-NEXT: init:
// CHECK-NEXT:   bid: b0
// CHECK-NEXT:   allocations:
// CHECK-NEXT:     %l0:i32:len
// CHECK-NEXT:     %l1:i32*:a 
// CHECK-NEXT:     %l2:i32:sum 
// CHECK-NEXT:     %l3:i32:i 
double average(int len, int a[10]) {
  // CHECK-LABEL: block b0:
  // CHECK-NEXT:   %b0:p0:i32:len 
  // CHECK-NEXT:   %b0:p1:i32*:a
  // CHECK-NEXT:   %b0:i0:unit = store %b0:p0:i32 %l0:i32* 
  // CHECK-NEXT:   %b0:i1:unit = store %b0:p1:i32* %l1:i32**
  int sum = 0;
  // CHECK-NEXT:   %b0:i2:unit = store 0:i32 %l2:i32*
  int i;

  // CHECK-NEXT:   %b0:i3:unit = store 0:i32 %l3:i32*
  // CHECK-NEXT:   j b1()
  for (i = 0; i < len; i++) {
    // CHECK-LABEL: block b1: 
    // CHECK-NEXT:   %b1:i0:i32 = load %l3:i32*
    // CHECK-NEXT:   %b1:i1:i32 = load %l0:i32*
    // CHECK-NEXT:   %b1:i2:i1 = cmp lt %b1:i0:i32 %b1:i1:i32
    // CHECK-NEXT:   %b1:i3:i32 = typecast %b1:i2:i1 to i32
    // CHECK-NEXT:   %b1:i4:i1 = cmp ne %b1:i3:i32 0:i32
    // CHECK-NEXT:   br %b1:i4:i1, b2(), b4()
    
    // CHECK-LABEL: block b2:
    // CHECK-NEXT:   %b2:i0:i32 = load %l2:i32*
    // CHECK-NEXT:   %b2:i1:i32* = load %l1:i32**
    // CHECK-NEXT:   %b2:i2:i32 = load %l3:i32*
    // CHECK-NEXT:   %b2:i3:i64 = typecast %b2:i2:i32 to i64
    // CHECK-NEXT:   %b2:i4:i64 = mul %b2:i3:i64 4:i64
    // CHECK-NEXT:   %b2:i5:i32* = getelementptr %b2:i1:i32* offset %b2:i4:i64
    // CHECK-NEXT:   %b2:i6:i32 = load %b2:i5:i32*
    // CHECK-NEXT:   %b2:i7:i32 = add %b2:i0:i32 %b2:i6:i32
    // CHECK-NEXT:   %b2:i8:unit = store %b2:i7:i32 %l2:i32*
    // CHECK-NEXT:   j b3()
    sum += a[i];

    // CHECK-LABEL: block b3:
    // CHECK-NEXT:   %b3:i0:i32 = load %l3:i32*
    // CHECK-NEXT:   %b3:i1:i32 = add %b3:i0:i32 1:i32
    // CHECK-NEXT:   %b3:i2:unit = store %b3:i1:i32 %l3:i32*
    // CHECK-NEXT:   j b1()
  }

  // CHECK-LABEL: block b4:
  // CHECK-NEXT:   %b4:i0:i32 = load %l2:i32*
  // CHECK-NEXT:   %b4:i1:f64 = typecast %b4:i0:i32 to f64
  // CHECK-NEXT:   %b4:i2:i32 = load %l0:i32*
  // CHECK-NEXT:   %b4:i3:f64 = typecast %b4:i2:i32 to f64
  // CHECK-NEXT:   %b4:i4:f64 = div %b4:i1:f64 %b4:i3:f64
  // CHECK-NEXT:   ret %b4:i4:f64
  return (double)sum / len;
}
// CHECK-NEXT: }

// CHECK: fun i32 @main () {
// CHECK-NEXT: init:
// CHECK-NEXT:   bid: b0
// CHECK-NEXT:   allocations:
// CHECK-NEXT:     %l0:[10 x i32]:a 
// CHECK-NEXT:     %l1:i32:len
// CHECK-NEXT:     %l2:i32:i
// CHECK-NEXT:     %l3:f32:avg
int main() {
  // CHECK-LABEL: block b0:
  int a[10];
  int len = 10;
  // CHECK-NEXT:   %b0:i0:unit = store 10:i32 %l1:i32*

  // CHECK-NEXT:   %b0:i1:unit = store 0:i32 %l2:i32*
  // CHECK-NEXT:   j b1()
  for (int i = 0; i < len; i++) {
    // CHECK-LABEL: block b1:
    // CHECK-NEXT:   %b1:i0:i32 = load %l2:i32*
    // CHECK-NEXT:   %b1:i1:i32 = load %l1:i32*
    // CHECK-NEXT:   %b1:i2:i1 = cmp lt %b1:i0:i32 %b1:i1:i32
    // CHECK-NEXT:   %b1:i3:i32 = typecast %b1:i2:i1 to i32
    // CHECK-NEXT:   %b1:i4:i1 = cmp ne %b1:i3:i32 0:i32
    // CHECK-NEXT:   br %b1:i4:i1, b2(), b4()
    
    // CHECK-LABEL: block b2:
    // CHECK-NEXT:   %b2:i0:i32* = getelementptr %l0:[10 x i32]* offset 0:i64
    // CHECK-NEXT:   %b2:i1:i32 = load %l2:i32*
    // CHECK-NEXT:   %b2:i2:i64 = typecast %b2:i1:i32 to i64
    // CHECK-NEXT:   %b2:i3:i64 = mul %b2:i2:i64 4:i64
    // CHECK-NEXT:   %b2:i4:i32* = getelementptr %b2:i0:i32* offset %b2:i3:i64
    // CHECK-NEXT:   %b2:i5:i32 = load %l2:i32*
    // CHECK-NEXT:   %b2:i6:unit = store %b2:i5:i32 %b2:i4:i32*
    // CHECK-NEXT:   j b3()
    a[i] = i;

    // CHECK-LABEL: block b3:
    // CHECK-NEXT:   %b3:i0:i32 = load %l2:i32*
    // CHECK-NEXT:   %b3:i1:i32 = add %b3:i0:i32 1:i32
    // CHECK-NEXT:   %b3:i2:unit = store %b3:i1:i32 %l2:i32*
    // CHECK-NEXT:   j b1()
  }

  // CHECK-LABEL: block b4:
  float avg = average(len, a);
  // CHECK-NEXT:   %b4:i0:i32 = load %l1:i32*
  // CHECK-NEXT:   %b4:i1:i32* = getelementptr %l0:[10 x i32]* offset 0:i64
  // CHECK-NEXT:   %b4:i2:f64 = call @average:[ret:f64 params:(i32, i32*)]*(%b4:i0:i32, %b4:i1:i32*)
  // CHECK-NEXT:   %b4:i3:f32 = typecast %b4:i2:f64 to f32
  // CHECK-NEXT:   %b4:i4:unit = store %b4:i3:f32 %l3:f32*

  // CHECK-NEXT:   %b4:i5:f32 = load %l3:f32* 
  // CHECK-NEXT:   %b4:i6:f64 = typecast %b4:i5:f32 to f64
  // CHECK-NEXT:   %b4:i7:i32 = call @is_close:[ret:i32 params:(f64, f64, f64, f64)]*(%b4:i6:f64, 4.5:f64, 0.0000000010000000000000001:f64, 0.10000000000000001:f64)
  // CHECK-NEXT:   ret %b4:i7:i32
  return is_close(avg, 4.5, 1e-09, 0.1);
}
// CHECK-NEXT: }
