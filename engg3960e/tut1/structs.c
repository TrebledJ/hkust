#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// 
//  syntax:
//  

struct MyStruct {
    int data;
    float more_data;
    int even_more_data[10]; //  We can even have arrays in structs!
};

/**
 +- MyStruct -------------------+
 |  int      data               |
 |  float    more_data          |
 |  int[10]  even_more_data     |
 +------------------------------+
 **/

typedef struct USTMember {
    char name[42];
    int ID;
    int age;
} USTMemberType;

/**
 +- USTMember ------------------+
 |  char[42]        name        |
 |  int             ID          |
 |  int             age         |
 +------------------------------+
 **/


//  Prints the contents of a USTMember.
void whodis(USTMemberType member) {
    printf("%s {%d} (age %d)\n", member.name, member.ID, member.age);
}


int main(void) {
    USTMemberType weishyy;

    //  Set Wei Shyy's data.
    strcpy(weishyy.name, "Wei Shyy");
    weishyy.ID = 1;
    weishyy.age = 65;

    printf("\n");
    whodis(weishyy);    //  Pass in the weishyy box to the function.
    printf("\n");

    //  We can perform various operations on data members, just like normal variables.
    weishyy.age++;
    printf("age: %d\n", weishyy.age);
    printf("name length: %lu\n", strlen(weishyy.name));
    
    int age = weishyy.age;
    int* p_age = &(weishyy.age);
    (*p_age)++;
    printf("age: %d\n", weishyy.age);
    printf("\n");
}
