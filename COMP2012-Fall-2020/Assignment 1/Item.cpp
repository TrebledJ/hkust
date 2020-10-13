#include "Item.h"


Item::Item(string name, int quantity)
    : name{name}
    , quantity{quantity}
{
}

string Item::getName() const
{
    return name;
}

int Item::getQuantity() const
{
    return quantity;
}

//add the specified amount to item quantity
//note: you can assume the input amount is always positive
void Item::add(int amount)
{
    quantity += amount;
}

//remove the specified amount from item quantity if possible
//if the removal is possible (i.e. amount is not more than the quantity), do the removal and return true
//otherwise, do nothing and return false
//note: you can assume the input amount is always positive
bool Item::remove(int amount)
{
    if (amount > quantity)
        return false;
    
    quantity -= amount;
    return true;
}