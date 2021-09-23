#include <iostream>
#include "inventory.h"

using namespace std;

//do NOT modify the given header files and the main*.cpp source files
//check the FAQ section also to see what you are disallowed to do
int main()
{
    cout << boolalpha << endl; //we want to see true/false instead of 1/0 in the console output
    cout << "Wow, so happy to work on a programming assignment again! Yes Yes YES! Let's play with inventories this time! ^_____^" << endl;
    cout << endl;

    cout << "===============================" << endl;
    Inventory a;
    a.addItem("Apple", 10);
    a.addItem("Banana", 20);
    a.addItem("Orange", 15);
    a.print();
    cout << "===============================" << endl;
    cout << "Deep copied inventories:" << endl;
    Inventory* copy1 = new Inventory(a);
    Inventory* copy2 = new Inventory(*copy1);
    copy1->addItem("Apple", 100);
    copy1->print();
    copy2->print();
    cout << "copy1's total used space = " << copy1->getTotalUsedSpace() << endl;
    cout << "run copy1->emptyInventory()..." << endl;
    copy1->emptyInventory();
    cout << "copy1's total used space = " << copy1->getTotalUsedSpace() << endl;
    cout << "===============================" << endl;
    cout << "Say goodbye to memory leak! :)" << endl;
    delete copy1;
    delete copy2;
    cout << "===============================" << endl;

    return 0;
}