#include <iostream>
#include "account.h"
using namespace std;

int main()
{
    Account ac(3000,3000,3000);
    Expenses* expense1 = new Expenses(-300, true, "02/11/2020", CLOTHS, CASH, "Dressing expenses");
    ac.addTransactionToLedger(expense1); 
}