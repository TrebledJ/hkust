// //  syntax:
// //  typedef <...> <new-type>;

// //  examples:
// typedef int my_int_type;
// typedef char keycode;

// int n = 1;
// my_int_type nn = 2;

// char letter = 'A';
// keycode k = 'A';


// #include <stdio.h>

// typedef int* my_int_pointer;

// int* p = NULL;
// my_int_pointer pp = NULL;



// //  Useful to know:
// typedef int (*my_int_function_pointer)(int, int);

// int my_add(int a, int b) { return a + b; }
// my_int_function_pointer fnptr = my_add;
// //  fnptr(1, 2) -> 3






























#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>


typedef double inttype;

inttype add(inttype a, inttype b) { return a + b; }
inttype sub(inttype a, inttype b) { return a - b; }
inttype mul(inttype a, inttype b) { return a * b; }

void print(const char* str, inttype x) {
    printf("%s: %d\n", str, x);
}

int main(void) {
    inttype a = 1;
    inttype b = 2;

    printf("\n");
    print("add", add(a, b));
    print("sub", sub(a, b));
    print("mul", mul(a, b));
    printf("\n");
}