#include <stdlib.h>
#include <stdio.h>

// 
//  syntax:
// 
//  enum AnEnum {
//      ItemOne,
//      ItemTwo,
//      ItemThree,
//  };
// 


typedef enum {
    Red = 10,
    Green,
    Blue,
    Yellow,
    Crimson = 10,
} ColorType;

ColorType get_color() { return Red; }


int main() {
    ColorType color = get_color();

    printf("\n");

    printf("Red: %d\n", Red);
    printf("Green: %d\n", Green);
    printf("Crimson: %d\n", Crimson);

    switch (color) {
    case Red: printf("Red!\n"); break;
    case Green: printf("Green!\n"); break;
    case Blue: printf("Blue!\n"); break;
    default: break;
    }

    if (color == Red || color == Green) {
        printf("What a lovely color!\n");
    }

    printf("\n");
}