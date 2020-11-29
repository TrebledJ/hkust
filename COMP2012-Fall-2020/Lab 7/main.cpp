#include <iostream>
#include "Engine.h"

#include "PoliceCar.h"
#include "Vehicle.h"
#include "DeliveryCar.h"

using namespace std;

int main() {

    //Vehicle
    cout << "Test A"<< endl;
    Vehicle *c = new Vehicle(12,Silver,12000);
    c->print();
    cout << endl;

    c->start();
    c->brake(1200);
    cout << endl;

    c->print();
    cout << endl;

    //PoliceCar
    PoliceCar *pc = new PoliceCar(8, Black, 100000, false);
    pc->print();
    cout << endl;

    pc->setInAction(true);
    pc->start();
    cout << endl;

    cout << "Current state of the police car: ";
    if(pc->getInAction())
        cout << "in action" << endl;
    else
        cout << "not in action" << endl;
    cout << endl;

    pc->brake(500);
    cout << endl;

    pc->setInAction(false);
    pc->print();

    delete c;
    c = pc;
    cout << endl;
    c->print();

    delete pc;
    cout << endl;

    // Vehicle
    cout << "Test B"<< endl;
    Vehicle *c2 = new Vehicle(6,Red,1500);
    c2->print();
    cout << endl;

    c2->start();
    c2->brake(20);
    cout << endl;

    c2->print();
    cout << endl;

    // DeliveryCar
    DeliveryCar *dc = new DeliveryCar(10,White,7500,20);
    dc->print();
    cout << endl;

    dc->start();
    dc->brake(120);
    cout << endl;

    dc->setChargePerMile(15);
    dc->print();
    cout << endl;

    delete c2;
    delete dc;
    

    return 0;
}
