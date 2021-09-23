#include <iostream>
#include "account.h"
using namespace std;

int main()
{
	Account ac(3000,3000,3000);
    Income* income1 = new Income(1000, true, "11/11/2020", POCKET_MONEY, CASH, "Pocket money from parents, yeahhh!"); 
    Income* income2 = new Income(300, false, "20/11/2020", INVEST_RETURNS, BANK, "Gained some money from ibond"); 
    Income* income3 = new Income(500, false, "20/12/2020", POCKET_MONEY, CASH, "Expected pocket money from parents"); 
    ac.addTransactionToLedger(income1);
    ac.addTransactionToLedger(income2);
    ac.addTransactionToLedger(income3);
    ac.getIncomeLedger()->printRecentNTrans(3);
}