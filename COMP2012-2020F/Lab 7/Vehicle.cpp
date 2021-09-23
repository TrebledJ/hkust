#include <iostream>
#include "Vehicle.h"
using namespace std;


Vehicle::Vehicle(int nc, Color color, int mileage)
	: Engine(nc)
	, m_color{color}
	, m_mileage{mileage}
{
}
Vehicle::~Vehicle()
{
	cout << endl;
	cout << "Calling Vehicle's destructor on the following vehicle:" << endl;
	Vehicle::print();
}

Color Vehicle::getColor() const { return m_color; }
int Vehicle::getMileage() const { return m_mileage; }
int Vehicle::getEngine() const { return getNumCylinder(); }

void Vehicle::start() {
	cout << "Vehicle with ";
	Start();
}

// Stop the engine and update the data member "mileage" by adding the traveled distance to it.
// Print the information.
// Please refer to the sample I/O.
// Use Engine's Stop();
void Vehicle::brake(int distance)
{
	cout << "Vehicle with ";
	Stop();
	cout << "Driving distance: " << distance << endl;
	m_mileage += distance;
}

//Print the information of the car. (Please refer to the I/O sample)
void Vehicle::print() const
{
	cout << "Engine: " << getEngine() << "   Color: " << getColor() << "   Current Miles: " << getMileage() << endl;
}