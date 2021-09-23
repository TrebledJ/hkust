
#ifndef POLICECAR_H_
#define POLICECAR_H_

#include "Vehicle.h"

class PoliceCar : public Vehicle
{
public:
	PoliceCar(int nc, Color c, int m, bool action);	//Constructor
	~PoliceCar(); //Destructor

    // Get PoliceCar's state
	bool getInAction() const; 

	// Set PoliceCar's state in action
	void setInAction(bool action);

	//Print the information of the PoliceCar. Please refer to the sample I/O.
	//Use Vehicle's print()
	virtual void print() const override;

private:
	bool m_inAction;	//Data member indicating whether the police car is in action.
};



#endif /* POLICECAR_H_ */
