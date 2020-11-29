#ifndef VEHICLE_H_
#define VEHICLE_H_

#include "Engine.h"


class Vehicle : private Engine	// Private inheritance
{
public:
	Vehicle(int nc, Color color, int mileage); // Constructor
	~Vehicle();

	Color getColor() const;
	int getMileage() const;
	int getEngine() const;

	// Start the engine and print the information.
	// Please refer to the sample I/O.
	// Use Engine's Start();
	void start();

	// Stop the engine and update the data member "mileage" by adding the traveled distance to it.
	// Print the information.
	// Please refer to the sample I/O.
	// Use Engine's Stop();
	void brake(int distance);

	//Todo: override print() 
	//Print the information of the car. (Please refer to the I/O sample)
	virtual void print() const override;

private:
	Color m_color;	// Color of the car
	int m_mileage;	// Total mileage
};


#endif /* VEHICLE_H_ */
