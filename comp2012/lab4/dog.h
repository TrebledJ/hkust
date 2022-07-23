#ifndef DOG_H
#define DOG_H

#include "animal.h"


class Dog : public Animal
{
public:
    Dog(const std::string& name);
    void talk() const;
    void eat() const;
    void bark() const;
};


#endif