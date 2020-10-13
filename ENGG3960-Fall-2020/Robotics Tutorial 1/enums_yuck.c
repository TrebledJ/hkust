#include <stdlib.h>
#include <stdio.h>


int get_color() { return 0; }

int main() {
    int color = get_color();

    printf("\n");

    switch (color) {
    case 0: printf("Red!\n"); break;
    case 1: printf("Green!\n"); break;
    case 2: printf("Blue!\n"); break;
    case 3: printf("Yellow!\n"); break;
    case 4: printf("Purple!\n"); break;
    default: break;
    }

    if (color == 0 || color == 1) {
        printf("What a lovely colour!\n");
    }
    
    printf("\n");
}