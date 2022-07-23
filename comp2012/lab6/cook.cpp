#include "cook.h"
#include <iostream>


Cook::Cook(const char* name, int hourly_wage, int hours_worked)
    : Employee(name)
    , m_hourly_wage{hourly_wage}
    , m_hours_worked{hours_worked}
{
}

Cook::~Cook()
{
    std::cout << "Cook Dtor" << std::endl;
}

void Cook::print_description() const
{
    Employee::print_description();
    std::cout << " Duty: Cook" << std::endl;
}

int Cook::salary() const
{
    return m_hourly_wage * m_hours_worked;
}