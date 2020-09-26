#ifndef FOODS_H
#define FOODS_H

class Bread
{
public:
  Bread();                   // Default constructor.
  Bread(const Bread &other); // Copy constructor.
  ~Bread();                  // Destructor.
};

class Meat
{
public:
  Meat();                  // Default constructor.
  Meat(const Meat &other); // Copy constructor.
  ~Meat();                 // Destructor.
};

class Burger
{
public:
  Burger(Bread *bread, Meat *meat); // General constructor, shallow copy the bread and meat.
  Burger(const Burger &other);      // Copy constructor, deep copy the bread and meat.
  ~Burger();                        // Destructor.

private:
  Bread *bread;
  Meat *meat;
};

#endif