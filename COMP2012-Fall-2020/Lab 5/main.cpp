#include <iostream>
#include "cat.h"
#include "fatcat.h"
#include "ultrafatcat.h"

using namespace std;

void printCatStatus(const Cat& cat) 
{
    cout << cat.getName() << "'s fullness is " << cat.getFullness() << "." << endl;
}

void playWithCat(Cat& cat)
{
    cat.play();
}

int main()
{
    {
        Cat c("Maru");
        cout << c.getName() << " eats a fish!" << endl;
        c.eatFish();
        printCatStatus(c);
        cout << c.getName() << " plays with you energetically!" << endl;
        playWithCat(c);
        printCatStatus(c);
    } //this bracket creates a scope, hence the Cat object would be destructed at the end of this scope here

    cout << "===============================================" << endl;

    {
        FatCat f("Fat Fat");
        cout << f.getName() << " eats a big fish!" << endl;
        f.eatBigFish();
        printCatStatus(f);
        cout << f.getName() << " plays with you!" << endl;
        playWithCat(f);
        printCatStatus(f);
    } //this bracket creates a scope, hence the FatCat object would be destructed at the end of this scope here

    cout << "===============================================" << endl;

    {
        UltraFatCat u("Big Fat");
        cout << u.getName() << " eats a fish!" << endl;
        u.eatFish();
        printCatStatus(u);
        cout << u.getName() << " eats a big fish too!" << endl;
        u.eatFishBuffet();
        printCatStatus(u);
        cout << u.getName() << " eats a delicious fish buffet!!!" << endl;
        u.eatFishBuffet();
        printCatStatus(u);
        cout << u.getName() << " plays with you lazily..." << endl;
        playWithCat(u);
        printCatStatus(u);
    } //this bracket creates a scope, hence the UltraFatCat object would be destructed at the end of this scope here

    cout << "===============================================" << endl;

    return 0;
}