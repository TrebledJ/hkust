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


    int testCase = 21;

    cout << endl << "=== Test case " << testCase << " ===" << endl << endl;

    if (testCase == 21) //test case derived from the provided skeleton main function
    {
        BST<int> bst1;
        bst1.add("e", 55);
        bst1.add("c", 33);
        bst1.add("g", 77);

        BST<int> bst2;
        bst2.add("d", 44);
        bst2.add("b", 22);
        bst2.add("f", 66);
        bst2.add("Z", 1);

        bst2.printTree(cout, 0);

        string start_1 = "b", end_1 = "g";
        list<int> *bst1ResultList = bst1.getBetweenRange(start_1, end_1);
        cout << "bst1.getBetweenRange Result Length: " << bst1ResultList->size() << endl;
        cout << "|";
        for (list<int>::iterator it = bst1ResultList->begin(); it != bst1ResultList->end(); ++it) {
            cout << *it << "|";
        }
        cout << endl;

        string start_2 = "Z", end_2 = "c";
        list<int> *bst2ResultList = bst2.getBetweenRange(start_2, end_2);
        cout << "bst2.getBetweenRange Result Length: " << bst2ResultList->size() << endl;
        cout << "|";
        for (list<int>::iterator it = bst2ResultList->begin(); it != bst2ResultList->end(); ++it) {
            cout << *it << "|";
        }
        cout << endl;
    } //end of test case

    return 0;;
}
