// do NOT modify this file
#ifndef __HEADER_GLOBAL__
#define __HEADER_GLOBAL__
#include <stdio.h>

extern const char *singleAtom[];
extern const char *doubleAtom[];
extern const char *errorMsg[];
extern const int singleAtomLength;
extern const int doubleAtomLength;

struct Atom;
struct Smiles;
enum VoidType;
struct ReturnType;

void *generateSmiles(void* smilesPtr);
void *count_atom(void* resultPtr1, void *resultPtr2);
void* max_atom(void *resultPtr);
void* min_atom(void* resultPtr);
void* count_all_atoms(void* resultPtr);
void* unique_atom(void* resultPtr);
void printElem(void *resultPtr);
void *addNum(void* resultPtr1, void *resultPtr2);

#endif