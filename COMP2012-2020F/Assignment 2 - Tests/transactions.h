#ifndef TRANSACTION_H
#define TRANSACTION_H

#include <string>
#include <iostream>
using namespace std;

enum TransactionCategory {SALARY, POCKET_MONEY, INVEST_RETURNS, GIFTS, HEALTH, FOOD, CLOTHS, RENTAL, TRANSPORT, ENTERTAINMENTS};
enum GoesTo {BANK, CASH, CREDIT_CARD};

class Transaction
{
public:
    Transaction(double amount, bool hasRealized, string date, string describe);
    virtual ~Transaction() = 0;
    
	void setAmount(double amount);
    double getAmount() const;
	
    // set if a transaction is realized or not. 
    void setHasRealized();
    // get the status of realization of a transaction
    bool getHasRealized();
	
	void setDate(string str);
	string getDate() const;	
	
	void setDescriptions(string str);
	string getDescriptions() const;
	
	virtual GoesTo getGoesTo() const = 0; 
	virtual TransactionCategory getCategory() const = 0;
    
    // these two functions are for operator overloading. 
    // Use printTransaction() in the operator overloading function to achieve polymorphism
    virtual ostream& printTransaction(ostream& out) const = 0;
    friend ostream& operator << (ostream &out, const Transaction &trans);
    
protected:
    double amount;
    // hasRealized: whether the actual transaction has taken place or not. 
    // just a label. The amount associated with a transaction still need to be added/deducted from the balance. 
    bool hasRealized;
	string date;
	string descriptions;
};

#endif