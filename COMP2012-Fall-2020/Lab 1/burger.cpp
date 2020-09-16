#include "burger.h"
#include <iostream>
using namespace std;
Burger::Burger() {
    b = new Bread;
    m = new Meat;
    cout << "Make Burger! " << endl << endl;
}

Burger::~Burger() {
    delete b;
    delete m;
}
