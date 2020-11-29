#ifndef EXPENSES_H
#define EXPENSES_H

#include "transactions.h"
#include <string>
#include <iostream>
using namespace std;

class Expenses: public Transaction
{
public:
    Expenses(double amount, bool hasRealized, string date, TransactionCategory ic, GoesTo gt,string description);
    ~Expenses() override;
    void setGoesTo(GoesTo gt);
	ostream& printTransaction(ostream& out) const override;
	TransactionCategory getCategory() const override;
    GoesTo getGoesTo() const override;
    
private:
    TransactionCategory sources;
    GoesTo sumToWhere;
};

#endif