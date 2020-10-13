#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>


// 
//  syntax:
// 
//  union MyUnion {
//      char c[4];
//      float f;
//      int i;
//  };
// 


typedef union {
    uint32_t i;
    float f;
} Number;


int main(void) {
    Number n;
    n.i = 1000000;                //  `n` now stores data from an int.
    n.f = 123.456;
    printf("i: %d\n", n.i); //  Type-punning. Re-interprets the bits (i.e. 1/0 data) stored `n` as a float.
    printf("f: %f\n", n.f);

    n.i = 3;
    printf("i: %d\n", n.i);
    printf("f: %f\n", n.f);
}
