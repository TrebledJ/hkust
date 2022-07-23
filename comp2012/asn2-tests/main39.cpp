#include <iostream>
#include "account.h"
using namespace std;

int main()
{
    Account ac(3000,3000,3000);
    Expenses* expense1 = new Expenses(-300, true, "02/11/2020", CLOTHS, CASH, "Dressing expenses");
    ac.addTransactionToLedger(expense1); 
    Expenses* expense2 = new Expenses(-200, false, "20/10/2020", ENTERTAINMENTS, CASH, "Video game <3"); 
    ac.addTransactionToLedger(expense2);
	ac.removeTransactionFromLedger(ac.getExpensesLedger(), 0);
}