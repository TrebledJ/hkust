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


    int testCase = 11;

    cout << endl << "=== Test case " << testCase << " ===" << endl << endl;

    if (testCase == 11) //test case derived from the provided skeleton main function
    {
        int (*fun)(int) = [](int n) { return n + 1; };
        int (*fun2)(int) = [](int n) { return n * 2; };

        int keys[] = {6, 1, 13, 20};
        int values[] = {66, 11, 133, 200};
        HashTable<int, int> *pht1 = new HashTable<int, int>(7, fun, fun2, Mode::LINEAR, 0.5);
        HashTable<int, int> *pht2 = new HashTable<int, int>(7, fun, fun2, Mode::LINEAR, 0.5);
        for (int i = 0; i < sizeof(keys) / sizeof(int); i++) {
            cout << "add {" << keys[i] << "," << values[i] << "}: collisions=" << pht1->add(keys[i], new int(values[i]))
                 << endl;
        }
        *pht2 = *pht1;
        cout << "add " << 28 << ": collisions=" << pht1->add(28, new int(2800)) << endl;
        cout << *pht1 << endl;
        cout << *pht2 << endl;
        delete pht1;
        delete pht2;
    } //end of test case

    return 0;
}