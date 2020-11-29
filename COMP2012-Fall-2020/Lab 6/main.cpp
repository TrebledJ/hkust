#include <iostream>

#include "cook.h"
#include "DeliveryMen.h"
#include "manager.h"

using namespace std;

int main(int argc, char **argv)
{
	Manager manager{"Bob", 20000};

	manager.hire(new Cook("Alice", 100, 80), 1);
	manager.hire(new DeliveryMen("Chris", 90, 150),2);
	manager.hire(new Cook("Bob", 90, 100),1);
	manager.hire(new DeliveryMen("Amy", 100, 50),2);
	manager.pay_salary();

	return 0;
}