#include "greatdane.h"
#include <iostream>

using namespace std;


GreatDane::GreatDane(const string& name)
    : Dog(name)
{
}

void GreatDane::bark() const
{
    cout << "WOOF WOOF!" << endl;
}