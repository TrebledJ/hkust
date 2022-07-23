#include "toy.h"
#include "cat.h"
#include <iostream>
using namespace std;


Toy::Toy(Cat* owner) //prints "[Owner's name]'s toy is created.". see sample output.
    : owner{owner}
{
    cout << owner->getName() << "'s toy is created." << endl;
}

Toy::~Toy()
{
    cout << owner->getName() << "'s toy is destroyed." << endl;
}