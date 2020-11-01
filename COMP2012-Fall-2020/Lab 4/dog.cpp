#include "dog.h"
#include <iostream>

using namespace std;


Dog::Dog(const string& name)
    : Animal(name)
{
}

void Dog::talk() const
{
    cout << "My name is " << getName() << "! I am a doggo!" << endl;
}

void Dog::eat() const
{
    cout << "GIMME THAT BONE!" << endl;
}

void Dog::bark() const 
{
    cout << "woof woof!" << endl;
}