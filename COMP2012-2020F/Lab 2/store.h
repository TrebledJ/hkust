#ifndef STORE_H
#define STORE_H

#include "foods.h"
#include <string>
using std::string; //use the std namespace just for string

class Store
{
public:
  Store(string owner, int maxNumBreads, int maxNumMeats, int maxNumBurgers); // TODO: General constructor.
  Store(const Store &other);                                   // TODO: Copy constructor, deep copy all shelves.
  ~Store();                                                    // TODO: Destructor.

  void cookBread();  // TODO: Cook a piece of bread by allocating a Bread object.
  void cookMeat();   // TODO: Cook a piece of meat by allocating a Meat object.
  void cookBurger(); // TODO: Cook a burger by allocating a Burger object.
                     // Since the construction of Burger needs bread and meat,
                     // it will consume one piece of bread and one piece of meat at the top of both shelves.
                     // The ownership of the top bread and top meat are transferred to the burger,
                     // which means the burger will destruct its bread and meat.

  void print() const; // print the store summary

private:
  string owner; //Owner; "string" is a very convenient C++ string class. Check our COMP2011 notes to learn more about it: https://course.cse.ust.hk/comp2011_2020S/notes/h.stlstr.pdf

  Bread **breadShelf;
  Meat **meatShelf;
  Burger **burgerShelf;

  int numBreads;  // Current number of breads.
  int numMeats;   // Current number of meats.
  int numBurgers; // Current number of burgers.

  int maxNumBreads;  // Maximal number of breads.
  int maxNumMeats;   // Maximal number of meats.
  int maxNumBurgers; // Maximal number of burgers.
};

#endif