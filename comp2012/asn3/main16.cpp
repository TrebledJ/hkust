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

    int testCase = 16;

    cout << endl << "=== Test case " << testCase << " ===" << endl << endl;

    if (testCase == 16) //additional tests of HashTable<K,T>::add and get, with rehashing
    {

        {
            cout << "Testing LibraryItem Class below..." << endl;
            LibraryItem item = LibraryItem("TestingLibraryItem", 20201106);
            item.displayInfo();
            cout << item << endl;
        }

        {
            cout << "Testing Book Class below..." << endl;
            Book item = Book("TestingBook", 20201106, "Library 1F Section A");
            item.displayInfo();
            cout << item << endl;
        }

    } //end of test case

    return 0;

}
