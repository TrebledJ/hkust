#include "Library.h"


Library::~Library()
{
    for (int i = 0; i < numberOfItems; i++)
        delete libraryItemArray[i];

    std::cout << "Library " << name << " Destructed" << std::endl;
}

void Library::addLibraryItem(LibraryItem* libraryItem)
{
    if (numberOfItems >= MAX_NUMBER_ITEM)
    {
        std::cout << "Number of Items exceeded" << std::endl;
        return;
    }

    // Add into array.
    libraryItemArray[numberOfItems++] = libraryItem;

    // Add item into other containers.
    hashTableIndex.add(libraryItem->getName(), libraryItem);
    bstIndex.add(std::to_string(libraryItem->getPublishedDate()), libraryItem);
}

LibraryItem* Library::searchLibraryItemByExactName(string name) const
{
    return hashTableIndex.get(name);
}

LibraryItem* Library::searchLibraryItemByExactPublishedDate(int targetDate) const
{
    return *bstIndex.get(std::to_string(targetDate));
}

list<LibraryItem*>* Library::searchLibraryItemByPublishedDateRange(int startDate, int endDate) const
{
    list<LibraryItem*>* items = bstIndex.getBetweenRange(std::to_string(startDate), std::to_string(endDate));
    if (!items->empty())
        return items;
        
    delete items;
    return nullptr;
}

bool Library::borrowItem(string name) const
{
    LibraryItem* item = hashTableIndex.get(name);
    if (!item)
    {
        std::cout << "Item " << name << " was not found" << std::endl;
        return false;
    }
    
    if (!item->getIsInStock())
    {
        std::cout << "Item " << name << " is not in stock currently" << std::endl;
        return false;
    }

    std::cout << "Borrowed Item " << name << std::endl;
    return true;
}

bool Library::returnItem(string name) const
{
    LibraryItem* item = hashTableIndex.get(name);
    if (!item)
    {
        std::cout << "Item " << name << " was not found" << std::endl;
        return false;
    }
    if (!item->getIsInStock())
        return false;
    
    std::cout << "Returned Item " << name << std::endl;
    return true;
}
