#ifndef DELIVERYMEN_H
#define DELIVERYMEN_H

#include "employee.h"


class DeliveryMen : public Employee
{
public:
    DeliveryMen(const char* name, int order_wage, int orders_worked);
    ~DeliveryMen();

    virtual void print_description() const override;

protected:
    virtual int salary() const override;

private:
    int m_order_wage;
    int m_orders_worked;
};

#endif
