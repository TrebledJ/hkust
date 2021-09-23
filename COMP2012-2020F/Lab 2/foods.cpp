#include "foods.h"

#include <iostream>

using namespace std;

// Bread

Bread::Bread()
{
    cout << "Bread Constructed!" << endl;
}

Bread::Bread(const Bread &other)
{
    cout << "Bread Copied!" << endl;
}

Bread::~Bread()
{
    cout << "Bread Destructed." << endl;
}

// Meat

Meat::Meat()
{
    cout << "Meat Constructed!" << endl;
}

Meat::Meat(const Meat &other)
{
    cout << "Meat Copied!" << endl;
}

Meat::~Meat()
{
    cout << "Meat Destructed." << endl;
}

// Burger

Burger::Burger(Bread *bread, Meat *meat)
    : bread(bread), meat(meat)
{
    cout << "Burger Constructed!" << endl;
}

Burger::Burger(const Burger &other)
{
    bread = other.bread ? new Bread(*other.bread) : nullptr;
    meat = other.meat ? new Meat(*other.meat) : nullptr;
    cout << "Burger Copied!" << endl;
}

Burger::~Burger()
{
    delete bread;
    delete meat;

    cout << "Burger Destructed." << endl;
}
