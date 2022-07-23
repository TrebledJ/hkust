#ifndef __CAT_H__
#define __CAT_H__

#include "toy.h"
#include <string>
using std::string; //uses "std" namespace for string

class Cat
{
public:
    Cat(string name); //prints "Cat constructor is called.". see sample output.
    ~Cat(); //prints "Cat destructor is called.". see sample output.
    Cat(const Cat&) = delete; //we disallow the usage of copy constructor (because we don't like copycats)
    string getName() const; //returns the name
    int getFullness() const; //returns the fullness
    void eatFish(); //increases fullness by 10
    void play(); //decreases fullness by 10
protected:
    int fullness; //how full the cat is: eating increases it while playing decreases it
private:
    string name; //cat's name
    Toy toy; //cat's toy; hint: you need to use MIL to initialize the Toy object
};

#endif // __CAT_H__