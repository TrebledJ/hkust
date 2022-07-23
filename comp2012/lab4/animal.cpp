#include "animal.h"
#include <iostream>

using namespace std;


Animal::Animal(const string& name)
    : m_name{name}
{
}

void Animal::talk() const
{
    cout << "My name is " << m_name <<  "! I am just an animal!" << endl;
}

void Animal::eat() const
{
    cout << "I eat something... nom nom nom" << endl;
}

const string& Animal::getName() const { return m_name; }