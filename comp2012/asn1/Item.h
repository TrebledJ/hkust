//do NOT modify this file
//do NOT submit this file

#include <string>
using namespace std;

class Item
{
public:
    //constructor
    //note: you can assume the input quantity is always positive
    Item(string name, int quantity);

    //return the item name
    string getName() const;

    //return the item quantity    
    int getQuantity() const;

    //add the specified amount to item quantity
    //note: you can assume the input amount is always positive
    void add(int amount);

    //remove the specified amount from item quantity if possible
    //if the removal is possible (i.e. amount is not more than the quantity), do the removal and return true
    //otherwise, do nothing and return false
    //note: you can assume the input amount is always positive
    bool remove(int amount);
private:
    string name; //item name
    int quantity; //item quantity
};