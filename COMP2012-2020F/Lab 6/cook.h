#ifndef COOK_H
#define COOK_H

#include "employee.h"


class Cook : public Employee
{
public:
    Cook(const char* name, int hourly_wage, int hours_worked);
    ~Cook();

    virtual void print_description() const override;

protected:
    virtual int salary() const override;

private:
    int m_hourly_wage;
    int m_hours_worked;
};


#endif
