#ifndef ACCOUNT_H
#define ACCOUNT_H

#include "ledger.h"

class Account
{
public:
	Account(double bank, double creditCard, double cash);
    ~Account();
    
    // print the balance of the cash, bank and credit card
    // refer to instruction page for output formats 
    void printBalance() const;
    
    // this function is to add a new transaction to the corresponding ledger
    // this function accepts new transaction as parameter. 
    // dynamically determine whether the transaction is Income or Expenses using dynamic_cast()
    // then add the transaction to the correct ledger (incomeLedger/expensesLedger)
    // remember to check the types of GoesTo for the transaction
    // and update the cashBalance/bankBalance/creditCardBalance
    // be careful about the positive sign and negative sign of Income and Expenses, refer to main() for their usage
    // for expenses, please check if cashBalance/bankBalance/creditCardBalance has enough balance before updates
    // refer to the instruction page/sample outputs for the proper outputs
    // make sure the amount goes to the correct cashBalance/bankBalance/creditCardBalance specified in transactions
	void addTransactionToLedger(Transaction* newTransact);	
 
    // this function removes a transaction from ledger by number (i.e. order of transaction, e.g. 0 represent the first transaction in the ledger)
    // First, check if there are any transactions in the ledger. If no, return without proceeding.
    // Then check if the numTransact is valid or not.
    // may make use of removeSingleTransaction() function.
    // similarly, check GoesTo type of the transaction before modifying the corresponding cashBalance/bankBalance/creditCardBalance.
    // be careful about the positive sign and negative sign in the update.
    // refer to the instruction page/sample outputs for the output
	void removeTransactionFromLedger(Ledger* led, int numTransact);
    
    // this function replaces an existing transaction from the ledger with a new one
    // note that both oldTrans and newTrans MUST be of the same type
    // dynamically determine whether the transaction is Income or Expenses using dynamic_cast()
    // and ensure both oldTrans and newTrans are of same types before proceeding
    // to update the ledger, remove a transaction from ledger and add a new one
    // the corresponding cashBalance/bankBalance/creditCardBalance in Account class should be changed accordingly. 
    // again, whenever modifications are needed in cashBalance/bankBalance/creditCardBalance, make sure correct SIGNS (+ or -) should be used
    // refer to the instruction page/sample outputs for outputs
	void updateLedger(Transaction* oldTrans, Transaction* newTrans);

    // a helper function as given
    void addBalance(int option, double amount);
    
    // get the private ledger data members
    Ledger* getIncomeLedger();
    Ledger* getExpensesLedger();
    
private:
	int maxNumTrans = 100;

	double bankBalance;
	double creditCardBalance;
	double cashBalance;
    Ledger* incomeLedger;
    Ledger* expensesLedger;    
};

#endif