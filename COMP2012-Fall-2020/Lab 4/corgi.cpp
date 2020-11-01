#include "corgi.h"
#include <iostream>

using namespace std;


Corgi::Corgi(const string& name)
    : Dog(name)
{
}

void Corgi::bark() const
{
    cout << "woooof woooof!" << endl;
}