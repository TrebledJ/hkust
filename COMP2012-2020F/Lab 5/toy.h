#ifndef __TOY_H__
#define __TOY_H__

class Cat; //forward declaration of Cat

class Toy
{
public:
    Toy(Cat* owner); //prints "[Owner's name]'s toy is created.". see sample output.
    ~Toy(); //prints "[Owner's name]'s toy is destroyed.". see sample output.
private:
    Cat* owner; //pointer to the owner (a cat)
};

#endif // __TOY_H__