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
    cout << "Let's do some tests on the Item class first." << endl;
    Item item("cookie", 3);
    cout << "I have " << item.getQuantity() << " " << item.getName() << "s!" << endl;
    cout << "Give me 2 more!" << endl;
    item.add(2);
    cout << "Now I have " << item.getQuantity() << " " << item.getName() << "s!" << endl;
    cout << "Take 4 away from me... result=" << item.remove(4) << endl;
    cout << "Now I have " << item.getQuantity() << " " << item.getName() << " left..." << endl;
    cout << "Try to take 4 more away from me... result=" << item.remove(4) << endl;
    cout << "Now I have " << item.getQuantity() << " " << item.getName() << " left..." << endl;
    cout << "===============================" << endl;

    cout << "OK, now we really play with the inventories!" << endl;
    Inventory a;
    cout << "Empty inventory:" << endl;
    a.print();
    cout << "===============================" << endl;

    cout << "Add 3 fruits:" << endl;
    a.addItem("Apple", 10);
    a.addItem("Banana", 20);
    a.addItem("Orange", 15);
    a.print();
    cout << "===============================" << endl;

    cout << "By the way, when we grade your work, we also check if the item indices are correct:" << endl;
    cout << "Apple's index = " << a.grading_getItemIndex("Apple") << endl;
    cout << "Banana's index = " << a.grading_getItemIndex("Banana") << endl;
    cout << "Orange's index = " << a.grading_getItemIndex("Orange") << endl;
    cout << "Grape's index = " << a.grading_getItemIndex("Grape") << endl;
    cout << "===============================" << endl;

    cout << "Anyway, let's make another inventory:" << endl;
    Inventory b;
    b.addItem("Banana", 5);
    b.addItem("Orange", 20);
    b.addItem("Grape", 30);
    b.print();
    cout << "===============================" << endl;

    cout << "Union:" << endl;
    Inventory* u = a.getUnion(b);
    u->print();
    delete u;
    cout << "===============================" << endl;

    cout << "Intersection:" << endl;
    Inventory* i = a.getIntersection(b);
    i->print();
    delete i;
    cout << "===============================" << endl;

    cout << "Difference:" << endl;
    Inventory* d = a.getDifference(b);
    d->print();
    delete d;
    cout << "===============================" << endl;

    cout << "Now let's add b and a together:" << endl;
    b.addInventory(a);
    cout << "a:" << endl;
    a.print();
    cout << "b:" << endl;
    b.print();
    cout << "===============================" << endl;

    cout << "Remove some bananas from a:" << endl;
    cout << "result=" << a.removeItem("Banana", 8) << endl;
    a.print();
    cout << "===============================" << endl;

    cout << "Remove too many bananas:" << endl;
    cout << "result=" << a.removeItem("Banana", 13) << endl;
    a.print();
    cout << "===============================" << endl;

    cout << "Remove all bananas:" << endl;
    cout << "result=" << a.removeItem("Banana", 12) << endl;
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
    cout << endl;
    cout << "Had enough fun with inventories... Peace! ;)" << endl;

    return 0;
}