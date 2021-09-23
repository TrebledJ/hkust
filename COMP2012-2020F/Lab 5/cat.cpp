#include "cat.h"
#include <iostream>
using namespace std;


Cat::Cat(string name)
    : fullness{0}
    , name{name}
    , toy{this}
{
    cout << "Cat constructor is called." << endl;
}

Cat::~Cat()
{
    cout << "Cat destructor is called." << endl;
}

string Cat::getName() const { return name; }
int Cat::getFullness() const { return fullness; }

void Cat::eatFish() { fullness += 10; }
void Cat::play() { fullness -= 10; }
