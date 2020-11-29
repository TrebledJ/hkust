#ifndef LEDGER_H
#define LEDGER_H
#include "transactions.h"
#include "income.h"
#include "expenses.h"

//remember, numTransact starts counting from 0, but currNumTrans starts from 1.

class Ledger
{
public:
    // make use of MAX_NUM_TRANS provided in ledger.cpp
	Ledger(int maxNumTrans, int currNumTrans);
    ~Ledger();
    
    // a function to add transaction to the ledger
    // remember to update the currNumTrans 
    void addTransaction(Transaction* newTransaction);
    
    // remove a transaction from ledger with number numTransact, which starts from 0
    // i.e. can remove any transaction with transaction number smaller than currNumTrans
    // remember to delete the object after removing the transaction from the ledger. 
    // also need to shift all the transactions forward, following the one just deleted 
    // e.g. transaction # 3 deleted, sequence becomes 0,1,2,4,5,..., etc. in the pointer array 
    // so there's no gap in the pointer array. 
    // remember, numTransact starts from 0.
    // Think about the relation between numTransact and currNumTrans, and reject invalid numTransact
    void removeSingleTransaction(int numTransact);

    // print every transaction in ledger sequentially. 
    // example outputs refer to instruction page/sample outputs 
    void printAllTransactions() const;
    
    // print the last N transactions in the ledger
    // the latest one got printed first, followed by the second latest one, etc. 
    void printRecentNTrans(int nTrans) const;
    
    
    // print all the realized/unrealized transactions from the ledger 
    void printRealizedTransactions(bool realized) const;
    
    // simply return the current number of transactions in the ledger. 
    int getCurrNumTrans() const;
    
    // return a transaction by numTransact
    // a helper function to be used in Account::removeTransactionFromLedger()
    Transaction* getTransactionByNum(int numTransact);
    // replace a transaction in ledger with another
    // firstly, identify the position of oldTrans in ledger
    // then replace it with newTrans
    // remember to do your memory management after replacement.
    // this function is to be called by Account::updateLedger() 
    // be careful about how to compare two double values. You may refer to lab 3 for hints.
    void updateTransactionInLedger(Transaction* oldTrans, Transaction* newTrans);
    
    
private:
    Transaction** allTransactions;
    int maxNumTrans;
    int currNumTrans;
};

#endif