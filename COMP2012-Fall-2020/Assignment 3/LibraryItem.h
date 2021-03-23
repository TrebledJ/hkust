#ifndef ASG3_LIBRARYITEM_H
#define ASG3_LIBRARYITEM_H

#include <iostream>
#include <string>

using namespace std;

class LibraryItem {
private:
    string name;
    int publishedDate;
    bool isInStock;
public:
    LibraryItem(string name, int publishedDate);

    LibraryItem(const LibraryItem &) {
        cout << "You are not allowed to call copy constructor of LibraryItem!!!" << endl;
    };

    virtual ~LibraryItem();

    int getPublishedDate() const;

    string getName() const;

    bool getIsInStock() const;

    void setIsInStock(bool isInStock);

    virtual void displayInfo() const;
};


ostream &operator<<(ostream &os, const LibraryItem &li);

#endif //ASG3_LIBRARYITEM_H
