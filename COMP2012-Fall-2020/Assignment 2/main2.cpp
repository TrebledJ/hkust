#include <iostream>
#include <string>
#include "account.h"

using std::string;
using namespace std;


int main()
{
    cout << "To award your hard work, it is a good idea to buy yourself some snacks to enjoy! :D" <<endl;
    cout << "So you have decided to spend around 150 dollars in CASH on snacks. " <<endl; 
    cout << endl; 
    
    Expenses* expense1 = new Expenses(-150, true, "11/10/2020", FOOD, CASH, "Snack expenses");
    cout << endl; 
    
    cout << "You have just seen the advertisement of a new video game and you feel interested in it. " << endl;
    cout << "You wish to buy but you know you should not be spending too much. " <<endl;
    cout << "Thus, you've decided to NOT buying the snacks, and use that money to buy the video game. " <<endl;
    cout << "The video game costs 200 dollars. You decide to pay in cash. Let's update the ledger ^_^." << endl; 
    cout << endl;

    Expenses* expense2 = new Expenses(-200, true, "20/10/2020", ENTERTAINMENTS, CASH, "Video game <3"); 
    cout << endl;    
    
    cout << "Now you may want to do some checking of the transactions. Let's check...." << endl;
    expense2->setAmount(-200);
    expense2->setGoesTo(CASH);
    expense2->setHasRealized();
    expense2->setDate("20/10/2020");
    cout << expense2->getGoesTo() << endl;
    cout << expense2->getCategory() << endl;
    cout << expense2->getAmount() << endl;
    cout << expense2->getDescriptions() << endl;
    cout << endl;    
    cout << *expense2 << endl;
    
    delete expense1;
    delete expense2;
    
    return 0;
} 