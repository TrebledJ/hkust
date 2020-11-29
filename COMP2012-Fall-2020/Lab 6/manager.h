#ifndef MANAGER_H
#define MANAGER_H

#include "employee.h"


class Manager : public Employee
{
public:
    Manager(const char* name, int base_salary);
    ~Manager();
    
    void hire(Employee *new_staff, int type);

    virtual void print_description() const override;
    void pay_salary() const;

protected:
    virtual int salary() const override;

private:
    static const int MAX_NUM_STAFF = 5;

    Employee* m_staff[MAX_NUM_STAFF];
    int m_num_staff = 0;

    int m_base_salary;
    int m_num_cook = 0;
    int m_num_deliverymen = 0;
};

#endif
