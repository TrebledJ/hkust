#include "Inventory.h"


Inventory::Inventory()
    : items{nullptr}
    , itemCount{0}
{
}

Inventory::Inventory(const Inventory& rhs)
    : items{rhs.itemCount > 0 ? new Item*[rhs.itemCount] : nullptr}
    , itemCount{rhs.itemCount}
{
    for (int i = 0; i < itemCount; ++i)
        items[i] = new Item(rhs.items[i]->getName(), rhs.items[i]->getQuantity());
}

//destructor
//you can make use of emptyInventory here
Inventory::~Inventory()
{
    emptyInventory();
}

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
void Inventory::addItem(string name, int quantity)
{
    int idx = getItemIndex(name);
    if (idx == -1)
    {
        //  Scenario A: Item doesn't exist, append to end.

        //  First create a new array.
        Item** newItems = new Item*[itemCount + 1];
        for (int i = 0; i < itemCount; ++i)
            newItems[i] = items[i];
        newItems[itemCount++] = new Item(name, quantity);

        //  Swap the new array and delete the old one.
        swap(items, newItems);
        delete[] newItems;
    }
    else
    {
        items[idx]->add(quantity);
    }
}

//remove the specificied quantity from the item of the specified name if possible
//if there is no item with the specified name, simply do nothing and return false
//otherwise, try to remove the given quantity from the item:
//    if the removal cannot be done (i.e. quantity to remove is more than the quantity of the item), do nothing and return false
//    if the removal can be done, then do the removal and return true
//one more thing: if, after removal, the quantity of the item becomes 0, you need to also remove the item completely from the items array (one of our sample output illustrates that)
//note: you can assume the input quantity is always positive
bool Inventory::removeItem(string name, int quantity)
{
    int idx = getItemIndex(name);
    if (idx == -1 || !items[idx]->remove(quantity))
        return false;

    if (items[idx]->getQuantity() == 0)
    {
        //  Remove item completely from inventory.
        delete items[idx];

        for (int i = idx + 1; i < itemCount; ++i)
            items[i - 1] = items[i];
        itemCount--;
    }
    return true;
}

//add another inventory to this inventory
//all items from another are added to this inventory
//existing items in this inventory are always kept in their current order at the front of the items array
//new items from another are added one by one in order to the end of this inventory's items array
//see sample output
void Inventory::addInventory(const Inventory& inv)
{
    for (int i = 0; i < inv.itemCount; ++i)
        addItem(inv.items[i]->getName(), inv.items[i]->getQuantity());
}

//empty the inventory
//delete all items, set itemCount to 0, set items to nullptr
//hint: you need to use delete [] for dynamic arrays, and delete for single dynamic objects
void Inventory::emptyInventory()
{
    if (itemCount > 0)
    {
        for (int i = 0; i < itemCount; ++i)
            delete items[i];
        
        delete[] items;
    }
    itemCount = 0;
    items = nullptr;
}

//return the quantity of the item with the given name
//simply return 0 if no item of that name is found
int Inventory::getQuantity(string name) const
{
    int i = getItemIndex(name);
    return i == -1 ? 0 : items[i]->getQuantity();
}

//return the sum of quantities of all items
//see sample output
int Inventory::getTotalUsedSpace() const
{
    int sum = 0;
    for (int i = 0; i < itemCount; ++i)
        sum += items[i]->getQuantity();
    return sum;
}

//create the union of this inventory (this) and another inventory (another), and return it
//union is defined as:
//1. only if an item exists in either this or another or both, it also exists in the union
//2. the quantity of an item in the union is the higher value of "item quantity in this" and "item quantity in another"
//existing items in this inventory are put in the resulting inventory first, and they are always kept in their original order 
//new items from another are added one by one in order to the end
//see sample output to help yourself understand 
//note: do NOT delete the resulting inventory that you return - it should be deleted by the function caller (e.g. the main function)
Inventory* Inventory::getUnion(const Inventory& rhs) const
{
    Inventory* inv = new Inventory(*this);
    for (int i = 0; i < rhs.itemCount; ++i)
    {
        int idx = getItemIndex(rhs.items[i]->getName());
        if (idx == -1)
            inv->addItem(rhs.items[i]->getName(), rhs.items[i]->getQuantity());
        else
        {
            inv->items[idx]->remove(inv->items[idx]->getQuantity());
            inv->items[idx]->add(std::max(items[idx]->getQuantity(), rhs.items[i]->getQuantity()));
        }
    }
    return inv;
}

//create the intersection of this inventory (this) and another inventory (another), and return it
//intersection is defined as:
//1. only if an item exists in both this and another, it also exists in the union
//2. the quantity of an item in the intersection is the lower value of "item quantity in this" and "item quantity in another"
//the item ordering follows the original item ordering in this inventory
//see sample output to help yourself understand 
//note: do NOT delete the resulting inventory that you return - it should be deleted by the function caller (e.g. the main function)
Inventory* Inventory::getIntersection(const Inventory& rhs) const
{
    Inventory* inv = new Inventory;
    for (int i = 0; i < itemCount; ++i)
    {
        int idx = rhs.getItemIndex(items[i]->getName());
        if (idx != -1)
            inv->addItem(items[i]->getName(), std::min(items[i]->getQuantity(), rhs.items[idx]->getQuantity()));
    }
    return inv;
}

//create the difference of this inventory (this) from another inventory (another), and return it
//difference is defined as:
//1. only if an item exists in this inventory and its resulting quantity (see next point) is larger than 0, it also exists in the difference
//2. the resulting quantity of the item = item quantity in this - item quantity in another
//the item ordering follows this inventory's 
//see sample output to help yourself understand 
//note: do NOT delete the resulting inventory that you return - it should be deleted by the function caller (e.g. the main function)
Inventory* Inventory::getDifference(const Inventory& rhs) const
{
    Inventory* inv = new Inventory(*this);
    for (int i = 0; i < rhs.itemCount; ++i)
    {
        const string name = rhs.items[i]->getName();
        const int idx = inv->getItemIndex(name);

        if (idx == -1)  //  Item in rhs not found in lhs.
            continue;

        const int q = inv->items[idx]->getQuantity();
        const int rq = rhs.items[i]->getQuantity();
        inv->removeItem(name, std::min(q, rq));
    }
    return inv;
}

//print the inventory
void Inventory::print() const
{
    cout << "{";
    for (int i = 0; i < itemCount; ++i)
    {
        cout << "\"" << items[i]->getName() << "\":" << items[i]->getQuantity();
        if (i < itemCount - 1)
            cout << ",";
    }
    cout << "}" << endl;
}

//return the index of the element in the items array that represents the item of the given name
//return -1 if the item is not found
int Inventory::getItemIndex(string name) const
{
    for (int i = 0; i < itemCount; ++i)
        if (items[i]->getName() == name)
            return i;
    return -1;
}