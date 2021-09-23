#include <stdio.h>
#include <math.h>


const float PI = 3.14159265f;

typedef float radians;
typedef float degrees;

degrees r2d(radians x) { return x * 180.0 / PI; }
radians d2r(degrees x) { return x * PI / 180.0; }

int main(void) {
    printf("\n");
    printf("%f\n", r2d(PI / 2));
    printf("%f\n", d2r(18.0f));
    printf("\n");
}