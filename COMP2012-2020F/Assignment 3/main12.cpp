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


    int testCase = 12;

    cout << endl << "=== Test case " << testCase << " ===" << endl << endl;

    if (testCase == 12) //additional test of BST<T>::isEmpty
    {
        BST<int> bst;
        cout << bst.isEmpty() << endl;
        cout << bst.add("a", 1) << endl;
        cout << bst.isEmpty() << endl;
    } //end of test case


    return 0;
}