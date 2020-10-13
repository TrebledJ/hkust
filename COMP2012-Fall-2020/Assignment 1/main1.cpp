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
    cout << "Now I have " << item.getQuantity() << " " << item.getName() << " left... T_T" << endl;
    cout << "Try to take 4 more away from me... result=" << item.remove(4) << endl;
    cout << "Now I have " << item.getQuantity() << " " << item.getName() << " left... T_T" << endl;
    cout << "===============================" << endl;

    cout << "OK, now we really play with the inventories!" << endl;
    Inventory a;
    cout << "Empty inventory:" << endl;
    a.print();
    cout << "===============================" << endl;

    return 0;
}