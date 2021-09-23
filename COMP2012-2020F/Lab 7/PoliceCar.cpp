#include "PoliceCar.h"
#include <iostream>
using namespace std;


PoliceCar::PoliceCar(int nc, Color c, int m, bool ia)
    : Vehicle(nc, c, m)
    , m_inAction{ia}
{
}
PoliceCar::~PoliceCar()
{
    cout << endl;
    cout << "Calling PoliceCar's destructor on the following police car:" << endl;
    print();
}

bool PoliceCar::getInAction() const { return m_inAction; }
void PoliceCar::setInAction(bool inAction) { m_inAction = inAction; }

void PoliceCar::print() const
{
    cout << "Information of the police car:" << endl;
    Vehicle::print();
    cout << "PoliceCar's state: " << (m_inAction ? "" : "not") << " in action" << endl;
}
