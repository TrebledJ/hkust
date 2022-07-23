#include <iostream>
using namespace std;

int main()
{
    /**
     * There are 3 errors below, 2 syntax errors.
     * After you have fixed all the errors, you can run the program and you should see
     * "Welcome! Please enter your lucky number between 1 to 9 inclusive: "
     *
     * Then you are prompted to input. You should have two different replies:
     * 1. When you enter a number such as 1, 2, 3, 4, ..., 9,
     * the reply is "You are a vigilant and thoughtful person!"
     * 2. When you enter other number, such as 0, 10, -1, 11, etc,
     * the reply is "You are a spontaneous and imaginative person!"
     */

    // Syntax error: a variable has to be defined before use
    int number;

    cout << "Welcome! " <<
            "Please enter your lucky number between 1 to 9 inclusive: " << endl;
   
    cin >> number;
    if ((number >= 1) && (number <= 9))
       cout << "You are a vigilant and thoughtful person!" << endl;
    else
       cout << "You are a spontaneous and imaginative person!" << endl;

    return 0;
}
