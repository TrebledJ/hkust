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


    int testCase = 10;

    cout << endl << "=== Test case " << testCase << " ===" << endl << endl;

    if (testCase == 10) //test case derived from the provided skeleton main function
    {
        int (*fun)(int) = [](int n) { return n + 1; };
        int (*fun2)(int) = [](int n) { return n * 2; };
        Mode modes[] = {Mode::LINEAR, Mode::QUADRATIC, Mode::DOUBLE};
        int keys[] = {6, 1, 13, 20, 99999};
        int values[] = {66, 11, 133, 200};
        Mode mode = modes[2]; //different mode for different test cases
        HashTable<int, int> ht(7, fun, fun2, mode, 0.5);

        for (int i = 0; i < sizeof(values) / sizeof(int); i++) {
            ht.add(keys[i], new int(values[i]));
        }

        cout << "current table:" << endl;
        cout << ht << endl;

        cout << "remove 1: " << ht.remove(1) << endl;
        cout << "remove 11: " << ht.remove(11) << endl;
        cout << ht << endl;

        cout << "add " << 28 << ": collisions=" << ht.add(28, new int(2800)) << endl;
        cout << ht << endl;
        cout << "add " << 27 << ": collisions=" << ht.add(27, new int(2700)) << endl;
        cout << ht << endl;
    } //end of test case



    return 0;
}
