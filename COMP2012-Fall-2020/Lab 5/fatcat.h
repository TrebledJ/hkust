#ifndef FATCAT_H
#define FATCAT_H

#include "cat.h"


class FatCat : public Cat
{
public:
    FatCat(string name);
    ~FatCat();
    void play();
    void eatBigFish();
};

#endif