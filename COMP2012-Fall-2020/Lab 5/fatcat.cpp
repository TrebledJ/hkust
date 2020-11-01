#include "fatcat.h"
#include <iostream>
using namespace std;


FatCat::FatCat(string name) : Cat(std::move(name)) { cout << "FatCat constructor is called." << endl; }
FatCat::~FatCat() { cout << "FatCat destructor is called." << endl; }

void FatCat::eatBigFish() { fullness += 20; }
void FatCat::play() { fullness -= 8; }
