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

    return 0;
}