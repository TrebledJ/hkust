#include "manager.h"
#include <iostream>


Manager::Manager(const char* name, int base_salary)
    : Employee(name)
    , m_base_salary{base_salary}
{
}

Manager::~Manager()
{
    for (int i = 0; i < m_num_staff; i++)
        delete m_staff[i];

    std::cout << "Manager Dtor" << std::endl;
}

void Manager::hire(Employee *new_staff, int type)
{
    m_staff[m_num_staff++] = new_staff;
    (type == 1 ? m_num_cook : m_num_deliverymen)++;
}

void Manager::print_description() const
{
    Employee::print_description();
    std::cout << " Duty: Manager" << std::endl;
}

void Manager::pay_salary() const
{
    for (int i = 0; i < m_num_staff; i++)
    {
        m_staff[i]->print_description();
        m_staff[i]->print_salary();
    }
    print_description();
    print_salary();
}

int Manager::salary() const
{
    return m_base_salary + 100 * m_num_cook + 50 * m_num_deliverymen;
}