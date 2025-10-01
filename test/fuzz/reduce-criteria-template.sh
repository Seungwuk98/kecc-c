#!/usr/bin/env bash

rm -f out*.txt

#ulimit -t 3000                                                                                                                                                                                             
#ulimit -v 2000000                                                                                                                                                                                          

if
  (! gcc -Wall -Wextra $REDUCED_C > $TEST_DIR/out_gcc.txt 2>&1)
then
  exit 1
fi

if
  [ $FUZZ_ARG = '-i' ] &&\
  (! clang -pedantic -Wall -Werror=strict-prototypes -c $REDUCED_C > $TEST_DIR/out_clang.txt 2>&1 ||\
  grep 'main-return-type' $TEST_DIR/out_clang.txt ||\
  grep 'conversions than data arguments' $TEST_DIR/out_clang.txt ||\
  grep 'int-conversion' $TEST_DIR/out_clang.txt ||\
  grep 'ordered comparison between pointer and zero' $TEST_DIR/out_clang.txt ||\
  grep 'ordered comparison between pointer and integer' $TEST_DIR/out_clang.txt ||\
  grep 'eliding middle term' $TEST_DIR/out_clang.txt ||\
  grep 'end of non-void function' $TEST_DIR/out_clang.txt ||\
  grep 'invalid in C99' $TEST_DIR/out_clang.txt ||\
  grep 'specifies type' $TEST_DIR/out_clang.txt ||\
  grep 'should return a value' $TEST_DIR/out_clang.txt ||\
  grep 'uninitialized' $TEST_DIR/out_clang.txt ||\
  grep 'incompatible pointer to' $TEST_DIR/out_clang.txt ||\
  grep 'incompatible integer to' $TEST_DIR/out_clang.txt ||\
  grep 'type specifier missing' $TEST_DIR/out_clang.txt ||\
  grep 'implicit-function-declaration' $TEST_DIR/out_clang.txt ||\
  grep 'infinite-recursion' $TEST_DIR/out_clang.txt ||\
  grep 'pointer-bool-conversion' $TEST_DIR/out_clang.txt ||\
  grep 'non-void function does not return a value' $TEST_DIR/out_clang.txt ||\
  grep 'too many arguments in call' $TEST_DIR/out_clang.txt ||\
  grep 'declaration does not declare anything' $TEST_DIR/out_clang.txt ||\
  grep 'not equal to a null pointer is always true' $TEST_DIR/out_clang.txt ||\
  grep 'empty struct is a GNU extension' $TEST_DIR/out_clang.txt ||\
  grep 'uninitialized' $TEST_DIR/out_gcc.txt ||\
  grep 'without a cast' $TEST_DIR/out_gcc.txt ||\
  grep 'control reaches end' $TEST_DIR/out_gcc.txt ||\
  grep 'return type defaults' $TEST_DIR/out_gcc.txt ||\
  grep 'cast from pointer to integer' $TEST_DIR/out_gcc.txt ||\
  grep 'useless type name in empty declaration' $TEST_DIR/out_gcc.txt ||\
  grep 'no semicolon at end' $TEST_DIR/out_gcc.txt ||\
  grep 'type defaults to' $TEST_DIR/out_gcc.txt ||\
  grep 'too few arguments for format' $TEST_DIR/out_gcc.txt ||\
  grep 'incompatible pointer' $TEST_DIR/out_gcc.txt ||\
  grep 'ordered comparison of pointer with integer' $TEST_DIR/out_gcc.txt ||\
  grep 'declaration does not declare anything' $TEST_DIR/out_gcc.txt ||\
  grep 'expects type' $TEST_DIR/out_gcc.txt ||\
  grep 'pointer from integer' $TEST_DIR/out_gcc.txt ||\
  grep 'incompatible implicit' $TEST_DIR/out_gcc.txt ||\
  grep 'excess elements in struct initializer' $TEST_DIR/out_gcc.txt ||\
  grep 'comparison between pointer and integer' $TEST_DIR/out_gcc.txt ||\
  grep 'division by zero' $TEST_DIR/out_gcc.txt)
then
  exit 1
fi

if
  $CLANG_ANALYZE &&\
  (! clang --analyze -c $REDUCED_C > $TEST_DIR/out_analyzer.txt 2>&1 ||\
  grep 'garbage value' $TEST_DIR/out_analyzer.txt)
then
  exit 1
fi

$FUZZ_BIN $REDUCED_C > $TEST_DIR/out_fuzz.txt 2>&1

grep $FUZZ_ERRMSG $TEST_DIR/out_fuzz.txt
