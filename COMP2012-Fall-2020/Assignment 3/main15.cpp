//do NOT submit this file.
//you may modify it to test your own test cases.
//however, make sure that your code can be compiled with the unmodified version of this file.

#include <iostream>
#include <string>
#include <list>
#include "LibraryItem.h"
#include "Book.h"
#include "BST.h"
#include "HashTable.h"
#include "Library.h"

using namespace std;

template<typename T>
std::ostream &operator<<(std::ostream &os, const BST<T> &t) {
    t.printTree(os);
    return os;
}


std::ostream &operator<<(std::ostream &os, const Library &library) {
    os << "Library: " << library.getName() << ", Number of Library Item: " << library.getNumberOfItems();
    return os;
}

template<typename A, typename B>
std::ostream &operator<<(std::ostream &os, const HashTable<A, B> &ht) {
    ht.print(os);
    return os;
}

std::ostream &operator<<(std::ostream &os, const Mode &m) {
    switch (m) {
        case Mode::LINEAR:
            os << "LINEAR";
            break;

        case Mode::DOUBLE:
            os << "DOUBLE";
            break;

        case Mode::QUADRATIC:
            os << "QUADRATIC";
            break;
    }
    return os;
}

int main() {
    cout << boolalpha;


    int testCase = 15;

    cout << endl << "=== Test case " << testCase << " ===" << endl << endl;

    if (testCase == 15) //additional tests of HashTable<K,T>::add and get, with rehashing
    {
        int (*fun)(int) = [](int n) { return n; };
        int (*fun2)(int) = [](int n) { return n * 3; };
        Mode modes[] = {Mode::LINEAR, Mode::QUADRATIC, Mode::DOUBLE};
        int keys[] = {1, 2, 3, 9, 17, 25};
        int values[] = {11, 22, 33, 99, 177, 255};
        Mode mode = modes[2]; //different mode for different test cases

        HashTable<int, int> ht(4, fun, fun2, mode, 0.5);
        cout << "empty table:" << endl;
        cout << ht << endl;
        for (int i = 0; i < sizeof(values) / sizeof(int); i++) {
            cout << "add {" << keys[i] << "," << values[i] << "}: collisions=" << ht.add(keys[i], new int(values[i]))
                 << endl;
            cout << ht << endl;
        }
    } //end of test case

    return 0;

}
