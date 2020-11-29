#include "account.h"
using namespace std;

int main()
{
    Ledger* led = new Ledger(100, 0);
	Income* income1 = new Income(2500, false, "20/10/2020", SALARY, BANK, "Part time wages");
	led->addTransaction(income1);
	delete led;	
}