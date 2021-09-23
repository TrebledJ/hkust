#include "account.h"
using namespace std;

int main()
{
    Ledger* led = new Ledger(100, 0);
    Expenses* expense1 = new Expenses(-300, true, "02/11/2020", CLOTHS, CASH, "Dressing expenses");
    Expenses* expense2 = new Expenses(-150, false, "13/11/2020", ENTERTAINMENTS, BANK, "Video game spending");
    Expenses* expense3 = new Expenses(-500, false, "30/10/2020", HEALTH, CASH, "Dental appointment");
    led->addTransaction(expense1); 
    led->addTransaction(expense2); 
    led->addTransaction(expense3); 
	Transaction* trans = led->getTransactionByNum(0);
    cout << *trans;
	delete led; 	
}