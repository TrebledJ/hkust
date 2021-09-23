#include <iostream>
#include "account.h"
using namespace std;

int main()
{
	Account ac(3000,3000,3000);
    Income* income1 = new Income(1000, true, "11/11/2020", POCKET_MONEY, CASH, "Pocket money from parents, yeahhh!"); 
    ac.addTransactionToLedger(income1);
    Expenses* expense1 = new Expenses(-200, false, "29/10/2020", ENTERTAINMENTS, BANK, "Movie ticket"); 
	
    ac.updateLedger(income1, expense1); 	
	delete expense1; 
}