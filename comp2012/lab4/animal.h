#ifndef ANIMAL_H
#define ANIMAL_H

#include <string>


class Animal
{
public:
    Animal(const std::string& m_name);

    void talk() const;
    void eat() const;
    
    const std::string& getName() const;

private:
    std::string m_name;
};


#endif