#include <iostream>
#include <string>
#include "account.h"

using std::string;
using namespace std;


int main()
{

    Expenses* expense1 = new Expenses(-3000, false, "30/10/2020", INVEST_RETURNS, BANK, "Investment Losses");
    expense1->setDate("28/11/2020");
	cout << expense1->getDate() << endl;
	
    delete expense1;

    return 0;
} 