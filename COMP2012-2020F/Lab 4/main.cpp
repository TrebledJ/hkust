#include <iostream>
#include "animal.h"
#include "dog.h"
#include "greatdane.h"
#include "corgi.h"
using namespace std;

void petDog(Dog doggo)
{
    cout << "You pet " << doggo.getName() << "..." << endl;
    cout << doggo.getName() << ": ";
    doggo.bark();
}

void petCorgi(const Corgi& corgi)
{
    cout << "You pet the fat belly of " << corgi.getName() << " the Corgi..." << endl;
    cout << corgi.getName() << ": ";
    corgi.bark();
}

int main()
{
    Animal a("Randy");
    a.talk();
    cout << "===============================" << endl;
    a.eat();
    cout << "===============================" << endl;

    Dog d("Warwick");
    d.talk();
    cout << "===============================" << endl;
    d.bark();
    cout << "===============================" << endl;
    d.eat();
    cout << "===============================" << endl;
    petDog(d);
    cout << "===============================" << endl;

    GreatDane g("Huge Huge");
    g.talk();
    cout << "===============================" << endl;
    g.bark();
    cout << "===============================" << endl;
    g.eat();
    cout << "===============================" << endl;

    Corgi c("Bomb Bomb");
    c.talk();
    cout << "===============================" << endl;
    c.bark();
    cout << "===============================" << endl;
    c.eat();
    cout << "===============================" << endl;
    petDog(c);
    cout << "===============================" << endl;
    petCorgi(c);
    cout << "===============================" << endl;

    string dogName = "Latte";
    petDog(dogName); 

    return 0;
}