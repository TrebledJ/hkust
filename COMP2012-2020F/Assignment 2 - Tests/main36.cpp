#include <iostream>
#include "account.h"
using namespace std;

int main()
{
	Account ac(3000,3000,3000);
    Income* income1 = new Income(1000, true, "11/11/2020", POCKET_MONEY, CASH, "Pocket money from parents, yeahhh!"); 
    Income* income2 = new Income(300, false, "20/11/2020", INVEST_RETURNS, BANK, "Gained some money from ibond"); 
    ac.addTransactionToLedger(income1);
    ac.addTransactionToLedger(income2);
}