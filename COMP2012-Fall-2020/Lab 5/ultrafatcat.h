#ifndef ULTRAFATCAT_H
#define ULTRAFATCAT_H

#include "fatcat.h"


class UltraFatCat : public FatCat
{
public:
    UltraFatCat(string name);
    ~UltraFatCat();
    void eatFishBuffet();
    void play();
};


#endif