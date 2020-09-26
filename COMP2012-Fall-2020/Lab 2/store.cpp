#include "store.h"

#include <iostream>
#include <algorithm>

using namespace std;

Store::Store(string owner, int maxNumBreads, int maxNumMeats, int maxNumBurgers)
    : owner{owner}
    , numBreads{0}
    , numMeats{0}
    , numBurgers{0}
    , maxNumBreads{maxNumBreads}
    , maxNumMeats{maxNumMeats}
    , maxNumBurgers{maxNumBurgers}
{
    // Allocate memory for meats, breads and burgers.
    breadShelf = new Bread*[maxNumBreads];
    meatShelf = new Meat*[maxNumMeats];
    burgerShelf = new Burger*[maxNumBurgers];

    fill(breadShelf, breadShelf + maxNumBreads, nullptr);
    fill(meatShelf, meatShelf + maxNumMeats, nullptr);
    fill(burgerShelf, burgerShelf + maxNumBurgers, nullptr);

    // Finish constructing with printing.
    cout << "Store Constructed!" << endl;
}

Store::Store(const Store &other)
    : owner{other.owner}
    , numBreads{other.numBreads}
    , numMeats{other.numMeats}
    , numBurgers{other.numBurgers}
    , maxNumBreads{other.maxNumBreads}
    , maxNumMeats{other.maxNumMeats}
    , maxNumBurgers{other.maxNumBurgers}
{
    // Clone the shelves.
    breadShelf = new Bread*[other.maxNumBreads];
    meatShelf = new Meat*[other.maxNumMeats];
    burgerShelf = new Burger*[other.maxNumBurgers];

    for (int i = 0; i < other.numBreads; ++i) breadShelf[i] = new Bread(*other.breadShelf[i]);
    for (int i = 0; i < other.numMeats; ++i) meatShelf[i] = new Meat(*other.meatShelf[i]);
    for (int i = 0; i < other.numBurgers; ++i) burgerShelf[i] = new Burger(*other.burgerShelf[i]);

    // Finish copying with printing.
    cout << "Store Copied!" << endl;
}

Store::~Store()
{
    // Destruct the shelves.
    for (int i = 0; i < numBreads; ++i) delete breadShelf[i];
    for (int i = 0; i < numMeats; ++i) delete meatShelf[i];
    for (int i = 0; i < numBurgers; ++i) delete burgerShelf[i];

    delete[] breadShelf;
    delete[] meatShelf;
    delete[] burgerShelf;
    
    // Finish destructing with printing.
    cout << owner << "'s store Destructed." << endl;
}

void Store::cookBread()
{
    if (numBreads >= maxNumBreads)
    {
        cout << "Error: Bread shelf is full, cooking failed." << endl;
    }
    else
    {
        // Cook a bread by allocating a Bread object.
        breadShelf[numBreads++] = new Bread;
    }
}

void Store::cookMeat()
{
    if (numMeats >= maxNumMeats)
    {
        cout << "Error: Meat shelf is full, cooking failed." << endl;
    }
    else
    {
        // Cook a meat by allocating a Meat object.
        meatShelf[numMeats++] = new Meat;
    }
}

void Store::cookBurger()
{
    if (numBurgers >= maxNumBurgers)
    {
        cout << "Error: Burger shelf is full, cooking failed." << endl;
    }
    else if (numMeats < 1 || numBreads < 1)
    {
        cout << "Error: Materials are insufficient for a burger." << endl;
    }
    else
    {
        // Cook a burger by allocating a Burger object,
        // Since the construction of Burger needs bread and meat,
        // it will consume one piece of bread and one piece of meat at the top of both shelves.
        // The ownership of the top bread and top meat are transferred to the burger,
        // which means the burger should destruct its bread and meat when it is being destructed.
        burgerShelf[numBurgers++] = new Burger(breadShelf[--numBreads], meatShelf[--numMeats]);
    }
}

void Store::print() const
{
    // Print the store summary according to the lab page description.
    // See the given code in destructor to learn how to print a string with cout.
    cout << owner << "'s store now has..." << endl;
    cout << "Bread: " << numBreads << "/" << maxNumBreads << endl;
    cout << "Meat: " << numMeats << "/" << maxNumMeats << endl;
    cout << "Burger: " << numBurgers << "/" << maxNumBurgers << endl;
}
