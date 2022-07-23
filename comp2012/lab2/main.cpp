#include <iostream>
#include <string>

#include "store.h"

using namespace std;

int main()
{
    cout << "1. Open the store for Johnson!" << endl;
    Store store("Johnson", 5, 5, 5);
    store.print();

    cout << "\n2. Try to cook a burger with only one piece of bread, failed." << endl;
    store.cookBread();
    store.print();
    store.cookBurger();

    cout << "\n3. Cook a piece of meat and then a burger, succeeded." << endl;
    store.cookMeat();
    store.print();
    store.cookBurger();
    store.print();

    cout << "\n4. Prepare the bread and meat again." << endl;
    store.cookBread();
    store.cookMeat();
    store.print();

    cout << "\n5. Clone the store, cook a burger and then destroy the cloned store." << endl;
    {
        Store clonedStore = store;
        clonedStore.print();
        clonedStore.cookBurger();
        clonedStore.print();
    } // "clonedStore" is destructed once the program runs out of this scope.

    cout << "\n6. Destroy the original store." << endl;
    store.print();
    cout << "Just before destruction... OK, let's do it." << endl;

    return 0;
} // "store" is destructed once the program runs out of this scope.