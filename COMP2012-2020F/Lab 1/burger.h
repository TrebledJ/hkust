#ifndef __BURGER_H
#define __BURGER_H

#include "meat.h"
#include "bread.h"

class Burger {
public:
    Burger();
    ~Burger();
private:
    Bread* b;
    Meat* m;
};

#endif