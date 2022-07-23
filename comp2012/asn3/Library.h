#ifndef ASG3_LIBRARY_H
#define ASG3_LIBRARY_H

#define MAX_NUMBER_ITEM 1000

#include "LibraryItem.h"
#include "BST.h"
#include "HashTable.h"

#include <string>

using std::string;


class Library {
private:
    string name;
    int numberOfItems;
    LibraryItem *libraryItemArray[MAX_NUMBER_ITEM]; // LibraryItem* array, items inside are dynamically allocated
    BST<LibraryItem *> bstIndex;  // Binary Search Tree indices for retrieving LibraryItem* by date range
    HashTable<string, LibraryItem> hashTableIndex; // Hash Table Indices for retrieving LibraryItem* by exact name
    //There should be exactly "numberOfItems" dynamically allocated LibraryItem
    //don't create

public:
    //constructor
    //use MIL to initialize the name, numberOfItems, the hashTableIndex and bstIndex
    Library(string name) : name(name),
                           hashTableIndex(
                                   7,
                                   [](string s) {
                                       return MAX_NUMBER_ITEM - ((int) s.length()) % MAX_NUMBER_ITEM;
                                   }, //provided hash function for hashing the string attribute
                                   [](int n) {
                                       return n * 2;
                                   },
                                   Mode::LINEAR, 0.5, true) {
        this->numberOfItems = 0;
    };

    //destructor
    //destruct the item
    //release all the dynamically allocated objects inside libraryItemArray even those they are allocated outside
    //output "Library {name} Destructed"
    ~Library();

    string getName() const { return this->name; }

    int getNumberOfItems() const { return this->numberOfItems; }

    //Add an unique library item
    //if the numberOfItem >= MAX_NUMBER_ITEM
    //print "Number of Items exceeded"
    //1. It will store the item into libraryItemArray (pointerwise assignment)
    //  P.S. We will destruct the LibraryItem one by one via libraryItemArray
    //2. Add the item into hashmap by name as key
    //3. Add the item into bst by the published date
    //P.S. We will add static allocated LibraryItem into Library
    //You should not create any new dynamically allocated object inside the function
    //Just assign the shallow copy
    void addLibraryItem(LibraryItem *libraryItem);

    //search the item by hashTableIndex
    //return the corresponding item if exist else return nullptr
    LibraryItem *searchLibraryItemByExactName(string name) const;

    //convert targetDate to string and search via that converted string
    //search the item via bstIndex
    LibraryItem *searchLibraryItemByExactPublishedDate(int targetDate) const;

    //convert both startDate and endDate to string
    //use the getBetweenRange function inside BST
    //if they exist return corresponding items
    //else return nullptr
    list<LibraryItem *> *searchLibraryItemByPublishedDateRange(int startDate, int endDate) const;

    //search the book via hashTableIndex
    //if not found print "Item {name} was not found" return false
    //if found and check isInStock true print "Borrowed Item {name}" and return true
    //else isInStock false print "Item {name} is not in stock currently" and return false
    bool borrowItem(string name) const;

    //search the book via hashTableIndex
    //if not found print "Item {name} was not found" return false
    //if found and check isInStock false print "Returned Item {name}" and return true
    bool returnItem(string name) const;

    void print() {
        for (int i = 0; i < numberOfItems; i++) {
            cout << *libraryItemArray[i] << endl;
        }
        cout << "BSTIndex" << endl;
        this->bstIndex.printTree(cout);
        cout << "HashTableIndex" << endl;
        this->hashTableIndex.print(cout);
    }
};

#endif //ASG3_LIBRARY_H
