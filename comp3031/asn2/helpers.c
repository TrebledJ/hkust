// do NOT modify this file
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdbool.h>
#include <string.h>
#include <limits.h>
#include "helpers.h"
#define MAXL 5000

void printElem(void *resultPtr);

const char *singleAtom[] = {"H", "B", "C", "N", "O", "F", "P", "S", "I", "b", "c", "n", "o", "s", "p", "K"};
const char *doubleAtom[] = {"He", "Li", "Be", "Ne", "Na", "Mg", "Al", "Si", "Ar", "Ca", "Cl", "Br"};
const int singleAtomLength = 16, doubleAtomLength = 12;
const char *errorMsg[] = {"Second parameter to function count_atom must be a single atom", 
    "Invalid results from other functions passed into function addNum",
    "Invalid parameter type, parameter must a Smiles generated from function generateSmiles"};

struct Atom{
    char name[3];
    int count;
};

struct Smiles{
    struct Atom *atoms;
    int atomsLength;
    int smilesLength;
};

enum VoidType{
    SMILES_T, ATOM_T, INT_T
};

struct ReturnType{
    void* ptr;
    enum VoidType type;
};


void *generateSmiles(void* smilesPtr){

    char *smiles = (char*)smilesPtr;
    struct Atom *atoms = (struct Atom*)malloc((singleAtomLength + doubleAtomLength) * sizeof(struct Atom));
    // struct Atom atoms[28];
    for(int i = 0; i != singleAtomLength; ++i)
    {
        strcpy(atoms[i].name, singleAtom[i]);
        atoms[i].count = 0;
    }
    for(int i = 0; i != doubleAtomLength; ++i)
    {
        int j = i + singleAtomLength;
        strcpy(atoms[j].name, doubleAtom[i]);
        atoms[j].count = 0;
    }

    char testAtom[3];
    for(int i = 0; i != strlen(smiles); ++i)
    {
        if(i != 0 && isdigit(smiles[i]) && smiles[i - 1] == 'H')
        {
            atoms[0].count += ((int)(smiles[i]) - 48 - 1);
            continue;
        }

        if(i != strlen(smiles) - 1)
        {
            testAtom[0] = smiles[i];
            testAtom[1] = smiles[i + 1];
            testAtom[2] = '\0';

            // printf("1. %s\n", testAtom);
            bool isDoubleAtom = false;
            for(int j = 0; j != doubleAtomLength; ++j)
            {
                if(strcmp(testAtom, doubleAtom[j]) == 0)
                {
                    ++(atoms[singleAtomLength + j].count);
                    isDoubleAtom = true;
                    ++i;
                    break;
                }
            }
            
            // printf("2. %d\n", isDoubleAtom);
            if(!isDoubleAtom)
            {
                for(int j = 0; j != singleAtomLength; ++j)
                {
                    testAtom[0] = smiles[i];
                    testAtom[1] = '\0';
                    if(strcmp(testAtom, singleAtom[j]) == 0)
                    {
                        ++(atoms[j].count);
                        break;
                    }
                }
            }
            // printf("3. %s\n\n", testAtom);
        }
        else
        {
            for(int j = 0; j != singleAtomLength; ++j)
            {
                testAtom[0] = smiles[i];
                testAtom[1] = '\0';
                if(strcmp(testAtom, singleAtom[j]) == 0)
                {
                    ++(atoms[j].count);
                    break;
                }
            }
        }
    }

    struct Smiles *smilesStruct = (struct Smiles*)malloc(sizeof(struct Smiles));
    smilesStruct->atoms = atoms;
    smilesStruct->atomsLength = singleAtomLength + doubleAtomLength;
    // smilesStruct->smilesLength = strlen(smiles);
    int atomCount = 0;
    for(int i = 0; i != (singleAtomLength + doubleAtomLength); ++i)
    {
        if(atoms[i].count != 0)
            atomCount += atoms[i].count;
    }
    smilesStruct->smilesLength = atomCount;

    struct ReturnType* resultPtr = (struct ReturnType*)malloc(sizeof(struct ReturnType));
    resultPtr->ptr = (void*)smilesStruct;
    resultPtr->type = SMILES_T;
    return (void*)resultPtr;
}



void *count_atom(void* resultPtr1, void *resultPtr2){
    
    bool invalidType = false;

    if(((struct ReturnType*)resultPtr1)->type != SMILES_T || ((struct ReturnType*)resultPtr2)->type != SMILES_T)
    {
        int *numPtr = (int*)malloc(sizeof(int));
        *numPtr = -3;
        struct ReturnType* resultPtr = (struct ReturnType*)malloc(sizeof(struct ReturnType));
        resultPtr->ptr = numPtr;
        resultPtr->type = INT_T;
        return (void*)resultPtr;
    }

    void* smilesStructPtr = ((struct ReturnType*)resultPtr1)->ptr;
    struct Smiles* smilesStruct = (struct Smiles*)smilesStructPtr;

    void* smilesStructPtr2 = ((struct ReturnType*)resultPtr2)->ptr;
    struct Smiles* smilesStruct2 = (struct Smiles*)smilesStructPtr2;
    char atomArr[3];
    atomArr[0] == '\0';
    bool atomFound = false;
    for(int i = 0; i != smilesStruct2->atomsLength; ++i)
    {
        if(smilesStruct2->atoms[i].count > 1 || (atomFound && smilesStruct2->atoms[i].count == 1))
        {
            atomArr[0] = '\0';
            break;
        }
        if(smilesStruct2->atoms[i].count == 1)
        {
            strcpy(atomArr, smilesStruct2->atoms[i].name);
            atomFound = true;
        }
    }

    if(atomArr[0] == '\0')
    {
        int *numPtr = (int*)malloc(sizeof(int));
        *numPtr = -1;
        struct ReturnType* resultPtr = (struct ReturnType*)malloc(sizeof(struct ReturnType));
        resultPtr->ptr = numPtr;
        resultPtr->type = INT_T;
        return (void*)resultPtr;
    }


    char *atom = atomArr;
    // char *atom = (char*)resultPtr2;
    if(atom[0] == '[')
    {   
        char newAtom[3];
        int i = 1;
        for(; atom[i] != ']'; ++i)
            newAtom[i - 1] = atom[i];
        atom = newAtom;
    }

    for(int i = 0; i != smilesStruct->atomsLength; ++i)
    {
        if(strcmp(smilesStruct->atoms[i].name, atom) == 0)
        {
            int *countPtr = (int*)malloc(sizeof(int));
            *countPtr = smilesStruct->atoms[i].count;
            struct ReturnType* resultPtr = (struct ReturnType*)malloc(sizeof(struct ReturnType));
            resultPtr->ptr = countPtr;
            resultPtr->type = INT_T;
            return (void*)resultPtr;
        }
    }
}


void* max_atom(void *resultPtr){

    if(((struct ReturnType*)resultPtr)->type != SMILES_T)
    {
        int *numPtr = (int*)malloc(sizeof(int));
        *numPtr = -3;
        struct ReturnType* resultPtr = (struct ReturnType*)malloc(sizeof(struct ReturnType));
        resultPtr->ptr = numPtr;
        resultPtr->type = INT_T;
        return (void*)resultPtr;
    }

    void* smilesStructPtr = ((struct ReturnType*)resultPtr)->ptr;
    struct Smiles* smilesStruct = (struct Smiles*)smilesStructPtr;
    int maxCount = 0;
    for(int i = 0; i != smilesStruct->atomsLength; ++i)
    {
        if(smilesStruct->atoms[i].count > maxCount)
            maxCount = smilesStruct->atoms[i].count;
    }
    int *countPtr = (int*)malloc(sizeof(int));
    *countPtr = maxCount;

    struct ReturnType* resultPtr1 = (struct ReturnType*)malloc(sizeof(struct ReturnType));
    resultPtr1->ptr = (void*)countPtr;
    resultPtr1->type = INT_T;
    return (void*)resultPtr1;
}


void* min_atom(void* resultPtr){

    if(((struct ReturnType*)resultPtr)->type != SMILES_T)
    {
        int *numPtr = (int*)malloc(sizeof(int));
        *numPtr = -3;
        struct ReturnType* resultPtr = (struct ReturnType*)malloc(sizeof(struct ReturnType));
        resultPtr->ptr = numPtr;
        resultPtr->type = INT_T;
        return (void*)resultPtr;
    }

    void* smilesStructPtr = ((struct ReturnType*)resultPtr)->ptr;
    struct Smiles* smilesStruct = (struct Smiles*)smilesStructPtr;
    int minCount = INT_MAX;
    for(int i = 0; i != smilesStruct->atomsLength; ++i)
    {
        if(smilesStruct->atoms[i].count < minCount && smilesStruct->atoms[i].count != 0)
            minCount = smilesStruct->atoms[i].count;
    }

    if(minCount == INT_MAX)
        minCount = 0;
    
    int *countPtr = (int*)malloc(sizeof(int));
    *countPtr = minCount;

    struct ReturnType* resultPtr1 = (struct ReturnType*)malloc(sizeof(struct ReturnType));
    resultPtr1->ptr = countPtr;
    resultPtr1->type = INT_T;
    return (void*)resultPtr1;
}


void* count_all_atoms(void* resultPtr){

    if(((struct ReturnType*)resultPtr)->type != SMILES_T)
    {
        int *numPtr = (int*)malloc(sizeof(int));
        *numPtr = -3;
        struct ReturnType* resultPtr = (struct ReturnType*)malloc(sizeof(struct ReturnType));
        resultPtr->ptr = numPtr;
        resultPtr->type = INT_T;
        return (void*)resultPtr;
    }

    void* smilesStructPtr = ((struct ReturnType*)resultPtr)->ptr;
    struct Smiles* smilesStruct = (struct Smiles*)smilesStructPtr;
    int totalCount = 0;
    for(int i = 0;i != smilesStruct->atomsLength; ++i)
    {
        if(smilesStruct->atoms[i].count != 0)
            totalCount += smilesStruct->atoms[i].count;
    }
    int *countPtr = (int*)malloc(sizeof(int));
    *countPtr = totalCount;

    struct ReturnType* resultPtr1 = (struct ReturnType*)malloc(sizeof(struct ReturnType));
    resultPtr1->ptr = countPtr;
    resultPtr1->type = INT_T;
    return (void*)resultPtr1;
}


void* unique_atom(void* resultPtr){

    if(((struct ReturnType*)resultPtr)->type != SMILES_T)
    {
        int *numPtr = (int*)malloc(sizeof(int));
        *numPtr = -3;
        struct ReturnType* resultPtr = (struct ReturnType*)malloc(sizeof(struct ReturnType));
        resultPtr->ptr = numPtr;
        resultPtr->type = INT_T;
        return (void*)resultPtr;
    }

    void* smilesStructPtr = ((struct ReturnType*)resultPtr)->ptr;
    struct Smiles* smilesStruct = (struct Smiles*)smilesStructPtr;
    int uniqueCount = 0;
    for(int i = 0; i != smilesStruct->atomsLength; ++i)
    {
        if(smilesStruct->atoms[i].count != 0)
            ++uniqueCount;
    }
    int *countPtr = (int*)malloc(sizeof(int));
    *countPtr = uniqueCount;

    struct ReturnType* resultPtr1 = (struct ReturnType*)malloc(sizeof(struct ReturnType));
    resultPtr1->ptr = countPtr;
    resultPtr1->type = INT_T;
    return (void*)resultPtr1;
}


void *addNum(void* resultPtr1, void *resultPtr2){
    
    int num1;
    enum VoidType type1 = ((struct ReturnType*)resultPtr1)->type;
    if(type1 == INT_T)
        num1 = *((int*)(((struct ReturnType*)resultPtr1)->ptr));
    else if(type1 == SMILES_T)
        num1 = ((struct Smiles*)(((struct ReturnType*)resultPtr1)->ptr))->smilesLength;
    else if(type1 == ATOM_T)
        num1 = ((struct Atom*)(((struct ReturnType*)resultPtr1)->ptr))->count;

    int num2;
    enum VoidType type2 = ((struct ReturnType*)resultPtr2)->type;
    if(type2 == INT_T)
        num2 = *((int*)(((struct ReturnType*)resultPtr2)->ptr));
    else if(type2 == SMILES_T)
        num2 = ((struct Smiles*)(((struct ReturnType*)resultPtr2)->ptr))->smilesLength;
    else if(type2 == ATOM_T)
        num2 = ((struct Atom*)(((struct ReturnType*)resultPtr2)->ptr))->count;

    if(num1 < 0 || num2 < 0)
    {
        int *numPtr = (int*)malloc(sizeof(int));
        *numPtr = -2;
        struct ReturnType* resultPtr = (struct ReturnType*)malloc(sizeof(struct ReturnType));
        resultPtr->ptr = numPtr;
        resultPtr->type = INT_T;
        return (void*)resultPtr; 
    }

    int result = num1 + num2;

    int *countPtr = (int*)malloc(sizeof(int));
    *countPtr = result;
    struct ReturnType* resultPtr = (struct ReturnType*)malloc(sizeof(struct ReturnType));
    resultPtr->ptr = countPtr;
    resultPtr->type = INT_T;
    return resultPtr;
}


void printElem(void *resultPtr){
    void* elementPtr = ((struct ReturnType*)resultPtr)->ptr;
    enum VoidType type = ((struct ReturnType*)resultPtr)->type;
    if(type == INT_T)
    {
        int num = *((int*)elementPtr);
        if(num >= 0)
            printf("%d\n", num);
        else
            printf("%s\n", errorMsg[-(num + 1)]);
    }
    else if(type == SMILES_T)
        printf("%d\n", ((struct Smiles*)elementPtr)->smilesLength);
    else if(type == ATOM_T)
        printf("%d\n", ((struct Atom*)elementPtr)->count);
}
