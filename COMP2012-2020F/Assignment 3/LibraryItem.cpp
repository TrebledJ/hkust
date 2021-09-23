#include "LibraryItem.h"
#include <iostream>

using namespace std;

LibraryItem::LibraryItem(string name, int publishedDate) {
    this->name = name;
    this->publishedDate = publishedDate;
    this->isInStock = true;
}

LibraryItem::~LibraryItem() {
}

int LibraryItem::getPublishedDate() const {
    return this->publishedDate;
};

string LibraryItem::getName() const {
    return this->name;
};

bool LibraryItem::getIsInStock() const {
    return this->isInStock;
};

void LibraryItem::setIsInStock(bool isInStock) {
    this->isInStock = isInStock;
};

void LibraryItem::displayInfo() const {
    string inStockMsg = (this->isInStock) ? "(in stock)" : "(not in stock)";
    cout << "Item: " << this->name
         << " published on " << this->publishedDate
         << inStockMsg << endl;
}

ostream &operator<<(ostream &os, const LibraryItem &li) {
    return (os << "Library Item: " << li.getName() << " " << li.getPublishedDate() << " (" << li.getIsInStock() << ")"
               << endl);
}
