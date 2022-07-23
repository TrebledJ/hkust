#include <stdlib.h>
#include <stdio.h>
#include <string.h>


typedef struct {
    char name[42];
    int ID;
    int age;
} USTMember;


/**
 +- USTMember ------------------+
 |  char[42]        name        |
 |  int             ID          |
 |  int             age         |
 +------------------------------+
 **/


void whodis(USTMember* member) {
    printf("%s {%d} (age %d)\n", (*member).name, (*member).ID, (*member).age);
    printf("%s {%d} (age %d)\n", member->name, member->ID, member->age);
}


int main(void) {
    USTMember weishyy;

    //  Set Wei Shyy's data.
    strcpy(weishyy.name, "Wei Shyy");
    weishyy.ID = 1;
    weishyy.age = 65;

    printf("\n");

    USTMember* p_weishyy = &weishyy;
    whodis(p_weishyy);
    printf("\n");
}
