#ifndef ASG3_BOOK_H
#define ASG3_BOOK_H

#include "LibraryItem.h"

class Book : public LibraryItem {
private:
    string physicalLocation; //Physical Location of the Book
public:
    Book(string name, int publishedDate, string physicalLocation);

    ~Book();

    void displayInfo() const;
};


#endif //ASG3_BOOK_H
