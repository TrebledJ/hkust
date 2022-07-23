#include "account.h"
using namespace std;

int main()
{
    Ledger* led = new Ledger(100, 0);
    Expenses* expense1 = new Expenses(-300, true, "02/11/2020", CLOTHS, CASH, "Dressing expenses");
	led->addTransaction(expense1);
	
	delete led;	
}