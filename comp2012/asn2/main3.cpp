#include <iostream>
#include <string>
#include "account.h"

using std::string;
using namespace std;


int main()
{
    cout << "Welcome to the financial management system! ^_^" <<endl;
    cout << "Let's start by initializing your account" <<endl;

    Account ac(1000,1000,1000);
    cout << endl;
    cout << "Now you have got a salary of 3000 dollars from your part-time job, paid directly to your bank account." <<endl; 
    cout << "Let's add this record into the income ledger! ^_^" <<endl;
    cout << endl; 

    cout << "Let test our legders by removing an arbitrary transaction from the empty ledger." << endl;
    ac.removeTransactionFromLedger(ac.getIncomeLedger(), 100);
    
    Income* income1 = new Income(5000, true, "11/10/2020", SALARY, BANK, "Part time salary");
    ac.addTransactionToLedger(income1);
    cout << endl; 
    
    cout << "As this is around the end of the year, and you receive annual bonus in form of double pay! Hurray!!" <<endl;
    cout << "Let's update our ledger! ^_^" <<endl; 
    cout << "Note: the double pay is calculated as = salary * 2" <<endl; 
    cout << endl;

    ac.addBalance(income1->getGoesTo(), -(income1->getAmount()));
    income1 = (*income1)*2;
    ac.addBalance(income1->getGoesTo(),income1->getAmount());
    
    cout << "Let's inspect the modified transaction. " << endl;
    ac.getIncomeLedger()->printRecentNTrans(1);
    
    cout << endl; 
    cout << "To award your hard work, it is a good idea to buy yourself some snacks to enjoy! :D" <<endl;
    cout << "So you have decided to spend around 150 dollars in CASH on snacks." <<endl; 
    cout << "Let's update our ledger! ^_^" <<endl; 
    cout << endl; 
    
    Expenses* expense1 = new Expenses(-150, true, "11/10/2020", FOOD, CASH, "Snack expenses"); 
    ac.addTransactionToLedger(expense1);   
    cout << endl; 
 
    
    cout << "You have realized next month will be your friend's birthday, so you decide to spare around 150 dollars to prepare him a present! " <<endl;
    cout << "This 150 dollars should now rest in your bank account for the time being. Nonetheless, you would like to record this down in ledger as unrealized expense as well. " <<endl;
    cout << endl; 
    Expenses* expense2 = new Expenses(-150, false, "11/10/2020", ENTERTAINMENTS, BANK, "Friend's birthday presents"); 
    ac.addTransactionToLedger(expense2);

    cout << endl; 
    cout << "OK. So now you want to take a look at the first 2 transactions at expense ledger. " << endl;
    ac.getExpensesLedger()->printRecentNTrans(2);
    cout << endl; 
    
    cout << "When around the end of month, you would like to have a summary of what your incomes and expenses incurred this month. This allows you to better inspect your financial status. " <<endl;       //traverse and print only realized transactions only.
    cout << "Let's print all incomes!" << endl;
    cout << endl; 
    ac.getIncomeLedger()->printAllTransactions();
    cout << endl; 
    cout << "Let's print all expenses!" << endl;
    ac.getExpensesLedger()->printAllTransactions();
    
    cout << endl; 
    cout << "But wait, some transactions are actually not realized yet...Let's find out what are they! ^_^" << endl;
    cout << "Retrieving unrealized incomes..." << endl; 
    ac.getIncomeLedger()->printRealizedTransactions(false);
    cout << "Retrieving unrealized expenses..." << endl;     
    ac.getExpensesLedger()->printRealizedTransactions(false);
    cout << endl; 
    
    cout << "You have just seen the advertisement of a new video game and you feel interested in it. " << endl;
    cout << "You wish to buy but you know you should not be spending too much. " <<endl;
    cout << "Thus, you've decided to NOT buying the snacks, and use that money to buy the video game. " <<endl;
    cout << "The video game costs 200 dollars. You decide to pay in cash. Let's update the ledger ^_^." << endl; 
    cout << endl;

    Expenses* expense3 = new Expenses(-200, true, "20/10/2020", ENTERTAINMENTS, CASH, "Video game <3"); 
    ac.updateLedger(expense1, expense3);
    cout << endl;

    cout << "Finally, let's take a look at what you have left. " <<endl;
    ac.printBalance();
    
    cout << endl; 
    cout << "After a while, when you are trying to play the newly bought video game, it does not work at all. " <<endl;
    cout << "So you decide to return it and ask back for full refund, as guaranteed on the packages." << endl;
    cout << "You need to remove this transaction from ledger as well. " <<endl;
    ac.removeTransactionFromLedger(ac.getExpensesLedger(), 0);
    cout << endl;
    cout << "In the following, we present you the error messages you need to include to prevent your program from running into troubles." << endl; 
    cout << "It also demonstrates some sample outputs of the functions." <<endl;
    cout << endl;
    
    cout << "Test # 1" << endl;
    Expenses* expense4 = new Expenses(-200, true, "20/10/2020", ENTERTAINMENTS, CASH, "Video game <3"); 
    ac.updateLedger(income1, expense4); 
    cout << endl;
    
    cout << "Test # 2" << endl;
    Income* income2 = new Income(5000, true, "11/10/2020", SALARY, BANK, "Part time salary");
    ac.addTransactionToLedger(income2);
    cout << endl;
    
    cout << "Test # 3" << endl;
    Expenses* expense6 = new Expenses(-300, true, "16/10/2020", FOOD, CREDIT_CARD, "Other expenses");
    ac.addTransactionToLedger(expense6);
    cout << endl;
    
    cout << "Test # 4" << endl;
    ac.removeTransactionFromLedger(ac.getIncomeLedger(), 100);
    cout << endl;
    
    cout << "Test # 5" << endl;
    Expenses* expense5 = new Expenses(-1000000000, true, "15/10/2020", FOOD, CASH, "Invalid expenses test"); 
    ac.addTransactionToLedger(expense5);
    cout << endl;

    cout << "Final clean ups" << endl; 
    ac.getIncomeLedger()->printAllTransactions();
    ac.getExpensesLedger()->printAllTransactions();
    
    delete expense4;
    
    return 0;
} 