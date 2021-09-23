#ifndef GREATDANE_H
#define GREATDANE_H

#include "dog.h"
#include <string>


class GreatDane : public Dog
{
public:
    GreatDane(const std::string& name);

    void bark() const;
};


#endif