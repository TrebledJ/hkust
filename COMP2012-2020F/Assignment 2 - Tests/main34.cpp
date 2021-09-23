#include "account.h"
using namespace std;

int main()
{
    Ledger* led = new Ledger(100, 0);
    Expenses* expense1 = new Expenses(-300, true, "02/11/2020", CLOTHS, CASH, "Dressing expenses");
    Expenses* expense2 = new Expenses(-150, false, "13/11/2020", ENTERTAINMENTS, BANK, "Video game spending"); 
    led->addTransaction(expense1);
	led->updateTransactionInLedger(expense1, expense2);
    
	delete led; 	
}