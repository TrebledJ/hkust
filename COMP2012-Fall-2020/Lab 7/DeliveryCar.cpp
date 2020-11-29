#include "DeliveryCar.h"
#include <iostream>
using namespace std;


DeliveryCar::DeliveryCar(int nc, Color c, int m, int charge)
    : Vehicle(nc, c, m)
    , m_chargePerMile{charge}
{
}
DeliveryCar::~DeliveryCar()
{
    cout << endl;
    cout << "Calling DeliveryCar's destructor on the following delivery car:" << endl;
    print();
}

int DeliveryCar::getChargePerMile() const { return m_chargePerMile; }
void DeliveryCar::setChargePerMile(int charge) { m_chargePerMile = charge; }

void DeliveryCar::start()
{
    Vehicle::start();
}
void DeliveryCar::brake(int d)
{
    Vehicle::brake(d);
}

void DeliveryCar::print() const
{
    cout << "Information of the delivery car:" << endl;
    Vehicle::print();
    cout << "DeliveryCar's charges per mile: " << m_chargePerMile << endl;
    cout << "DeliveryCar's calculated charges: " << m_chargePerMile * getMileage() << endl;
}
