#include <iostream>
#include "account.h"
using namespace std;

int main()
{
    Account ac(3000,3000,3000);
    Expenses* expense1 = new Expenses(-300000, true, "02/11/2020", CLOTHS, CASH, "Dressing expenses");
    Expenses* expense2 = new Expenses(-150, false, "13/11/2020", ENTERTAINMENTS, BANK, "Video game spending");
    Expenses* expense3 = new Expenses(-500, false, "30/10/2020", HEALTH, CASH, "Dental appointment"); 
    ac.addTransactionToLedger(expense1); 
    ac.addTransactionToLedger(expense2); 
    ac.addTransactionToLedger(expense3); 
    ac.getExpensesLedger()->printRealizedTransactions(true);
    ac.getExpensesLedger()->printRealizedTransactions(false);
}