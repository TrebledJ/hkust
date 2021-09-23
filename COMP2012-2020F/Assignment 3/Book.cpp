//
// Created by Arthur on 6/11/2020.
//

#include "Book.h"

Book::Book(string name, int publishedDate, string physicalLocation) : LibraryItem(name, publishedDate)
{
    this->physicalLocation = physicalLocation;
}

Book::~Book()
{
    cout << "Book " << this->getName() << " Destructed" << endl;
}

void Book::displayInfo() const
{
    LibraryItem::displayInfo();
    cout << "Book: " << this->physicalLocation << endl;
}
