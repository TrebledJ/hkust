#include <iostream>
#include <string>
#include "account.h"

using std::string;
using namespace std;


int main()
{
    cout << "Now you have got a salary of 3000 dollars from your part-time job, paid directly to your bank account." <<endl; 
    cout << "Let's initialize a record! ^_^" <<endl;
    cout << endl; 

    Income* income1 = new Income(5000, true, "11/10/2020", SALARY, BANK, "Part time salary");

    cout << endl; 
    
    cout << "As this is around the end of the year, and you receive annual bonus in form of double pay! Hurray!!" <<endl;
    cout << "Let's update our ledger! ^_^" <<endl; 
    cout << "Note: the double pay is calculated as = salary * 2" <<endl; 
    cout << endl;

    income1 = (*income1)*2;
    cout << endl;

    cout << "Now you expect to get 1000 pocket money in CASH. This is a UNREALIZED income but you want to record it first." << endl;
    Income* income2 = new Income(1000, false, "11/10/2020", POCKET_MONEY, CASH, "Pocket Money");
    cout << endl;
    cout << "You want to make some changes after you have got 1200 dollars." << endl;
    income1->setAmount(1200);
    income1->setGoesTo(CASH);
    income1->setHasRealized();
    income1->setDate("20/10/2020");
    cout << income1->getGoesTo() << endl;
    cout << income1->getCategory() << endl;
    cout << income1->getAmount() << endl;
    cout << income1->getDescriptions() << endl;
    cout << endl;
    cout << *income1 << endl; 
    
    delete income1;
    delete income2;

    return 0;
} 