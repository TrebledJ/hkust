#include "DeliveryMen.h"
#include <iostream>


DeliveryMen::DeliveryMen(const char* name, int order_wage, int orders_worked)
    : Employee(name)
    , m_order_wage{order_wage}
    , m_orders_worked{orders_worked}
{
}

DeliveryMen::~DeliveryMen()
{
    std::cout << "DeliveryMen Dtor" << std::endl;
}

void DeliveryMen::print_description() const
{
    Employee::print_description();
    std::cout << " Duty: DeliveryMen" << std::endl;
}

int DeliveryMen::salary() const
{
    return m_order_wage * m_orders_worked;
}
