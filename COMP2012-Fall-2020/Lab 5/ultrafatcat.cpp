#include "ultrafatcat.h"
#include <iostream>
using namespace std;


UltraFatCat::UltraFatCat(string name) : FatCat{std::move(name)} { cout << "UltraFatCat constructor is called." << endl; }
UltraFatCat::~UltraFatCat() { cout << "UltraFatCat destructor is called." << endl; }

void UltraFatCat::eatFishBuffet() { fullness += 30; }
void UltraFatCat::play() { fullness -= 5; }