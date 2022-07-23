#ifndef CORGI_H
#define CORGI_H

#include "dog.h"
#include <string>


class Corgi : public Dog
{
public:
    Corgi(const std::string& name);

    void bark() const;
};


#endif