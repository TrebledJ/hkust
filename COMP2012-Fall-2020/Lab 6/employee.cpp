#include <iostream>
#include <cstring>

#include "employee.h"
#include <iostream>


Employee::Employee(const char* name)
{
	// 1. Allocate dynamic memory for private member "name"
	// Hint: strlen() function is useful here
	m_name = new char[strlen(name) + 1];

	// 2. Copy the name using strcpy
	strcpy(m_name, name);
}

Employee::~Employee()
{
	std::cout << "Employee Dtor: " << m_name << std::endl;

	// Free dynamically allocated memory
	delete[] m_name;
}

void Employee::print_description() const
{
	std::cout << "Employee: " << m_name;
}

void Employee::print_salary() const
{
	std::cout << "Salary: " << salary() << std::endl;
}