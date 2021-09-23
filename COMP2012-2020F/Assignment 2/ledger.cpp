#include "ledger.h"
#include <cmath>


namespace
{
    bool operator== (Transaction& lhs, Transaction& rhs)
    {
        return fabs(lhs.getAmount() - rhs.getAmount()) < 0.0001
                && lhs.getHasRealized() == rhs.getHasRealized()
                && lhs.getDate() == rhs.getDate();
    }
}


// make use of MAX_NUM_TRANS provided in ledger.cpp
Ledger::Ledger(int maxNumTrans, int currNumTrans)   //  What's the currNumTrans param for???
    : maxNumTrans{maxNumTrans}
    , currNumTrans{0}
{
    allTransactions = new Transaction*[maxNumTrans];
    for (int i = 0; i < maxNumTrans; i++)
        allTransactions[i] = nullptr;
}

Ledger::~Ledger()
{
    for (int i = 0; i < maxNumTrans; i++)
        delete allTransactions[i];
    delete[] allTransactions;
}

// a function to add transaction to the ledger
// remember to update the currNumTrans 
void Ledger::addTransaction(Transaction* newTransaction)
{
    if (currNumTrans >= maxNumTrans)
    {
        std::cout << "Memory is full. Failed to add transaction!" << std::endl;
        return;
    }
    allTransactions[currNumTrans++] = newTransaction;
}

// remove a transaction from ledger with number numTransact, which starts from 0
// i.e. can remove any transaction with transaction number smaller than currNumTrans
// remember to delete the object after removing the transaction from the ledger. 
// also need to shift all the transactions forward, following the one just deleted 
// e.g. transaction # 3 deleted, sequence becomes 0,1,2,4,5,..., etc. in the pointer array 
// so there's no gap in the pointer array. 
// remember, numTransact starts from 0.
// Think about the relation between numTransact and currNumTrans, and reject invalid numTransact
void Ledger::removeSingleTransaction(int numTransact)
{
    if (currNumTrans == 0)
    {
        std::cout << "Sorry, no transaction to remove!" << std::endl;
        std::cout << "Invalid transaction number!" << std::endl;
        return;
    }
    if (numTransact < 0 || numTransact >= currNumTrans)
    {
        std::cout << "Invalid transaction number!" << std::endl;
        return;
    }

    delete allTransactions[numTransact];
    for (int i = numTransact+1; i < currNumTrans; i++)
        allTransactions[i-1] = allTransactions[i];
    
    //  Shrink the real container.
    allTransactions[currNumTrans--] = nullptr;
}

// print every transaction in ledger sequentially. 
// example outputs refer to instruction page/sample outputs 
void Ledger::printAllTransactions() const
{
    if (currNumTrans == 0)
    {
        std::cout << "Sorry, no transaction to print!" << std::endl;
        return;
    }

    for (int i = 0; i < currNumTrans; i++)
        std::cout << *allTransactions[i];
}

// print the last N transactions in the ledger
// the latest one got printed first, followed by the second latest one, etc. 
void Ledger::printRecentNTrans(int nTrans) const
{
    const int end = std::max(currNumTrans - nTrans, 0);
    if (nTrans < 0 || end > currNumTrans-1)
    {
        std::cout << "Sorry, no transaction to print!" << std::endl;
        return;
    }

    for (int i = currNumTrans-1; i >= end; i--)
        std::cout << *allTransactions[i];
}

// print all the realized/unrealized transactions from the ledger 
void Ledger::printRealizedTransactions(bool realized) const
{
    if (currNumTrans == 0)
    {
        std::cout << "Sorry, no transaction to remove!" << std::endl;
        return;
    }
    int count = 0;
    for (int i = 0; i < currNumTrans; i++)
        if (allTransactions[i]->getHasRealized() == realized)
        {
            std::cout << *allTransactions[i];
            count++;
        }
    if (count == 0)
        std::cout << "There is no " << (realized ? "" : "un") << "realized transaction." << std::endl;
}

// simply return the current number of transactions in the ledger. 
int Ledger::getCurrNumTrans() const
{
    return currNumTrans;
}

// return a transaction by numTransact
// a helper function to be used in Account::removeTransactionFromLedger()
Transaction* Ledger::getTransactionByNum(int numTransact)
{
    if (currNumTrans == 0)
    {
        std::cout << "Sorry, no transaction to retrieve!" << std::endl;
        return nullptr;
    }
    if (numTransact < 0 || numTransact >= currNumTrans)
    {
        std::cout << "Wrong transaction number! Cannot retrieve transactions!" << std::endl;
        return nullptr;
    }
    return allTransactions[numTransact];
}

// replace a transaction in ledger with another
// firstly, identify the position of oldTrans in ledger
// then replace it with newTrans
// remember to do your memory management after replacement.
// this function is to be called by Account::updateLedger() 
// be careful about how to compare two double values. You may refer to lab 3 for hints.
void Ledger::updateTransactionInLedger(Transaction* oldTrans, Transaction* newTrans)
{
    for (int i = 0; i < currNumTrans; i++)
        if (*allTransactions[i] == *oldTrans)
        {
            delete allTransactions[i];
            allTransactions[i] = newTrans;
            return;
       }
    std::cout << "No matching old transaction found!" << std::endl;
}
