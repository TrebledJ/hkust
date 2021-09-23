#ifndef INCOME_H
#define INCOME_H

#include "transactions.h"
#include <string>
#include <iostream>
using namespace std;

class Income: public Transaction
{
public:
    Income(double amount, bool hasRealized, string date, TransactionCategory ic, GoesTo gt, string description);
    ~Income() override;
    void setGoesTo(GoesTo gt);
	ostream& printTransaction(ostream& out) const override;
	TransactionCategory getCategory() const override;
    GoesTo getGoesTo() const override;
    
    Income* operator*(int salaryMultiplier);
    
private:
    TransactionCategory sources;
    GoesTo sumToWhere;
};

#endif