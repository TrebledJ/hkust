#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>


enum Action {
    EAT_ICE_CREAM = 1,
    STUDY,
    WORK_OUT,
    VISIT_RELATIVES,
    PLAY,
    SLEEP
};


//  Print the options.
void print_options(void) {
    printf("\n");
    printf("What would you like to do today?\n");
    printf(" 1. Eat ice cream\n");
    printf(" 2. Study\n");
    printf(" 3. Work out\n");
    printf(" 4. Visit relatives\n");
    printf(" 5. Play\n");
    printf(" 6. Sleep\n");
}

//  Returns an integer input.
enum Action get_choice(void) {
    int choice;
    scanf(" %d", &choice);
    return (enum Action)choice;
}

int main(void) {
    bool done = false;

    //  Infinite loop, so that options will keep printing.
    while (!done) {
        print_options();
        enum Action choice = get_choice();

        //  Print out some text depending on choice.
        switch (choice)
        {
        case EAT_ICE_CREAM: 
            printf("Yummy!\n");
            break;
        
        case STUDY:
            printf("You studied for a bit.\n");
            break;

        case WORK_OUT:
            printf("You worked out for a bit.\n");
            break;

        case VISIT_RELATIVES:
            printf("You visited your grandmother's abode.\n");
            break;

        case PLAY:
            printf("You played with Thomas the Tank Engine toy set.\n");
            break;

        case SLEEP:
            printf("Good night... zzz...\n");
            break;

        default:
            printf("(Option was not recognised.)\n");
            continue;   //  Go back to the start of the while-loop.
        }

        //  Print some additional text.
        if (choice == STUDY || choice == WORK_OUT) {
            printf("You feel stronger and more intelligent.\n");
        }

        if (choice == SLEEP) {
            break;  //  Quit the loop when sleep is chosen.
        }
    }

    return 0;
}