//do NOT modify this file
//do NOT submit this file

#include <iostream>
#include "Item.h"
using namespace std;

class Inventory
{
public:
    //default constructor
    //set itemCount to 0
    //set items to nullptr
    Inventory();

    //deep copy constructor
    Inventory(const Inventory& another);

    //destructor
    //you can make use of emptyInventory here
    ~Inventory(); 

    //add an item with the specified name and quantity
    //there are two scenarios:
    //(A) the item doesn't exist in the current inventory
    //(B) the item is already in the current inventory
    //for scenario (A), you need to create a new item with the specified name and quantity
    //you must store the new item at the very end of the items array
    //the items array must be just big enough to contain all items
    //therefore you must increase the size of the current items array (if any) by 1
    //remember to update the itemCount
    //for scenario (B), just add the quantity to the existing item
    //hint: check what member functions you can use for the Item objects
    //note: if you follow our instructions, your item ordering should be exactly the same as ours
    //note: you can assume the input quantity is always positive
    void addItem(string name, int quantity);

    //remove the specificied quantity from the item of the specified name if possible
    //if there is no item with the specified name, simply do nothing and return false
    //otherwise, try to remove the given quantity from the item:
    //    if the removal cannot be done (i.e. quantity to remove is more than the quantity of the item), do nothing and return false
    //    if the removal can be done, then do the removal and return true
    //one more thing: if, after removal, the quantity of the item becomes 0, you need to also remove the item completely from the items array (one of our sample output illustrates that)
    //note: you can assume the input quantity is always positive
    bool removeItem(string name, int quantity);

    //add another inventory to this inventory
    //all items from another are added to this inventory
    //existing items in this inventory are always kept in their current order at the front of the items array
    //new items from another are added one by one in order to the end of this inventory's items array
    //see sample output
    void addInventory(const Inventory& another);

    //empty the inventory
    //delete all items, set itemCount to 0, set items to nullptr
    //hint: you need to use delete [] for dynamic arrays, and delete for single dynamic objects
    void emptyInventory();

    //return the quantity of the item with the given name
    //simply return 0 if no item of that name is found
    int getQuantity(string name) const;

    //return the sum of quantities of all items
    //see sample output
    int getTotalUsedSpace() const;

    //create the union of this inventory (this) and another inventory (another), and return it
    //union is defined as:
    //1. only if an item exists in either this or another or both, it also exists in the union
    //2. the quantity of an item in the union is the higher value of "item quantity in this" and "item quantity in another"
    //existing items in this inventory are put in the resulting inventory first, and they are always kept in their original order 
    //new items from another are added one by one in order to the end
    //see sample output to help yourself understand 
    //note: do NOT delete the resulting inventory that you return - it should be deleted by the function caller (e.g. the main function)
    Inventory* getUnion(const Inventory& another) const;

    //create the intersection of this inventory (this) and another inventory (another), and return it
    //intersection is defined as:
    //1. only if an item exists in both this and another, it also exists in the union
    //2. the quantity of an item in the intersection is the lower value of "item quantity in this" and "item quantity in another"
    //the item ordering follows the original item ordering in this inventory
    //see sample output to help yourself understand 
    //note: do NOT delete the resulting inventory that you return - it should be deleted by the function caller (e.g. the main function)
    Inventory* getIntersection(const Inventory& another) const;

    //create the difference of this inventory (this) from another inventory (another), and return it
    //difference is defined as:
    //1. only if an item exists in this inventory and its resulting quantity (see next point) is larger than 0, it also exists in the difference
    //2. the resulting quantity of the item = item quantity in this - item quantity in another
    //the item ordering follows this inventory's 
    //see sample output to help yourself understand 
    //note: do NOT delete the resulting inventory that you return - it should be deleted by the function caller (e.g. the main function)
    Inventory* getDifference(const Inventory& another) const;

    //print the inventory
    //see the webpage for description
    //see the sample output for examples
    //use the online grader Zinc to verify your output format well before the submission deadline
    void print() const;

    //don't mind this...
    //this isn't really part of the inventory
    //it is used for grading purposes only
    //to expose the private member for grading
    int grading_getItemIndex(string name) const
    {
        return getItemIndex(name);
    }

private:
    Item** items; //dynamic array of Item pointers; it stores all the items in this inventory; it should always be just big enough to contain all items
    int itemCount; //count of element in the items array; it is also the exact size of the items array since the items array has no empty slot

    //return the index of the element in the items array that represents the item of the given name
    //return -1 if the item is not found
    int getItemIndex(string name) const; 
};