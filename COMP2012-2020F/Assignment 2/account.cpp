#include "account.h"
#include <iostream>


const int MAX_NUM_TRANS = 100;
string to_string(GoesTo);


Account::Account(double bank, double creditCard, double cash)
    : bankBalance{bank}
    , creditCardBalance{creditCard}
    , cashBalance{cash}
{
    int maxTrans = MAX_NUM_TRANS;
    int curTrans = MAX_NUM_TRANS;   //  Idk why?
    
    incomeLedger = new Ledger(maxTrans, curTrans);
    expensesLedger = new Ledger(maxTrans, curTrans);

    std::cout << "Initialization of account succeed! There are two ledger: Income and Expenses." << std::endl;
    std::cout << "==========Account Summary==========" << std::endl;
    std::cout << "Maximum #. transactions allowed for each ledger: " << maxTrans << std::endl;
    std::cout << "Current #. transactions allowed for each ledger: " << curTrans << std::endl;
    std::cout << "Current bank balance: " << bankBalance << std::endl;
    std::cout << "Current credit card balance: " << creditCardBalance << std::endl;
    std::cout << "Current cash balance: " << cashBalance << std::endl;
}

Account::~Account()
{
    delete incomeLedger;
    delete expensesLedger;
    std::cout << "All account records deleted." << std::endl;
}
    
// print the balance of the cash, bank and credit card
// refer to instruction page for output formats 
void Account::printBalance() const
{
    std::cout << "Printing Balances" << std::endl;
    std::cout << "Bank: " << bankBalance << std::endl;
    std::cout << "Credit Card: " << creditCardBalance << std::endl;
    std::cout << "Cash: " << cashBalance << std::endl;
}

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
void Account::addTransactionToLedger(Transaction* newTransact)
{
    const bool isIncome = dynamic_cast<Income*>(newTransact);
    const GoesTo gt = newTransact->getGoesTo();

    //  Check balance.
    const double balance = (gt == BANK ? bankBalance : gt == CASH ? cashBalance : creditCardBalance);
    if (!isIncome && balance < -newTransact->getAmount())
    {
        std::cout << "Oh no! Not enough money to spend! :(" << std::endl;
        delete newTransact; //  What the douche?
        return;
    }

    //  Check ledger.
    Ledger*& ledger = (isIncome ? incomeLedger : expensesLedger);
    if (ledger->getCurrNumTrans() >= MAX_NUM_TRANS)
    {
        std::cout << "Memory is full. Failed to add transaction!" << std::endl; //  Ugh.
        return;
    }

    //  Update balance.
    addBalance(gt, newTransact->getAmount());
    std::cout << newTransact->getAmount() << " added into " << to_string(gt) << "." << std::endl;

    std::cout << "Transaction amount " << newTransact->getAmount() << " added" << std::endl;

    //  Add to ledger.
    ledger->addTransaction(newTransact);
}

// this function removes a transaction from ledger by number (i.e. order of transaction, e.g. 0 represent the first transaction in the ledger)
// First, check if there are any transactions in the ledger. If no, return without proceeding.
// Then check if the numTransact is valid or not.
// may make use of removeSingleTransaction() function.
// similarly, check GoesTo type of the transaction before modifying the corresponding cashBalance/bankBalance/creditCardBalance.
// be careful about the positive sign and negative sign in the update.
// refer to the instruction page/sample outputs for the output
void Account::removeTransactionFromLedger(Ledger* led, int numTransact)
{
    if (led->getCurrNumTrans() == 0)
    {
        std::cout << "Error! No transactions in this ledger!" << std::endl;
        return;
    }
    if (numTransact < 0 || numTransact >= led->getCurrNumTrans())
    {
        std::cout << "Wrong transaction number! Cannot retrieve transactions!" << std::endl;
        return;
    }
    Transaction* trans = led->getTransactionByNum(numTransact);
    addBalance(trans->getGoesTo(), trans->getAmount());
    led->removeSingleTransaction(numTransact);
    std::cout << "An transaction has been removed successfully." << std::endl;
}

// this function replaces an existing transaction from the ledger with a new one
// note that both oldTrans and newTrans MUST be of the same type
// dynamically determine whether the transaction is Income or Expenses using dynamic_cast()
// and ensure both oldTrans and newTrans are of same types before proceeding
// to update the ledger, remove a transaction from ledger and add a new one
// the corresponding cashBalance/bankBalance/creditCardBalance in Account class should be changed accordingly. 
// again, whenever modifications are needed in cashBalance/bankBalance/creditCardBalance, make sure correct SIGNS (+ or -) should be used
// refer to the instruction page/sample outputs for outputs
void Account::updateLedger(Transaction* oldTrans, Transaction* newTrans)
{
    const bool isOldIncome = dynamic_cast<Income*>(oldTrans);
    const bool isNewIncome = dynamic_cast<Income*>(newTrans);
    if (isOldIncome != isNewIncome)
    {
        std::cout << "New and Old transaction types not matching!" << std::endl;
        std::cout << "Fail to insert records!" << std::endl;
        return;
    }

    //  TODO: check new balance has enough money?
    addBalance(oldTrans->getGoesTo(), -oldTrans->getAmount());
    addBalance(newTrans->getGoesTo(), newTrans->getAmount());

    Ledger*& ledger = (isOldIncome ? incomeLedger : expensesLedger);
    ledger->updateTransactionInLedger(oldTrans, newTrans);

    std::cout << "Successfully updated." << std::endl;
}

// a helper function as given
void Account::addBalance(int option, double amount)
{
    if (option < 0 || option > 2)
    {
        cout << "Wrong balance option number!" << endl;
        return;
    }

    double& balance = (option == BANK ? bankBalance : option == CASH ? cashBalance : creditCardBalance);
    balance += amount;
}

// get the private ledger data members
Ledger* Account::getIncomeLedger()
{
    return incomeLedger;
}
Ledger* Account::getExpensesLedger()
{
    return expensesLedger;
}