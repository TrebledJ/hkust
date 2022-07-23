#include <iostream>
using namespace std;

// you are required to implement a cross breing game,
// don't need to check the invalidation

int main(int argc, char** argv)
{
    cout << "we want go there/-----------------------------------\\\n"
            << "        /      //-----------------------------------\\\\\n"
            << "Adam  Bob     ///----        Bridge Game        ----\\\\\\\n"
            << "Clair Dave   ///----                             ----\\\\\\\n"
            << endl;
    cout << "> Adam, Bob, Clair and Dave want to cross the bridge\n"
            "> The bridge can take no more than two people at the same time.\n"
            "> Beside, it's very dark so the man who is crossing the bridge has to use the only torch\n"
            "> Characters Introduction: \n"
            "    <Adam> and <Bob> each takes *1* and *2* mins to cross the bridge\n"
            "    <Clair> needs *5* mins to cross the bridge\n"
            "    <Dave> will need *10* mins to make it\n"
            "> You are a clever passer-by and come up with an idea that help them decide how to cross the bridge...\n"
            "> !To control who should cross the bridge, input the first letter of only one person at a time.\n"
            ">  if you input more than one, characters after the first one will be ignored." << endl;
    cout << "\n\n !!!Game Start!!!\n\n\n" << endl;

    /*********************************************
     * here begins the main logic
     **********************************************/

    // location, true for left side
    bool adam = true, bob = true, clair = true, dave = true, torch = true;
    // bridge crossing time
    const int ADAM_TIME = 1, BOB_TIME = 2, CLAIR_TIME = 5, DAVE_TIME = 10;
    // Optimal time
    const int OPTIMAL_TIME = 17;
    // time used by player
    int time_used = 0;

    while (true)
    {
        cout << ">>> current state:\n";
        cout << (adam ? "(Adam)" : "      ") << (bob ? "(Bob)" : "     ")
                     << (clair ? "(Clair)" : "       ") << (dave ? "(Dave)" : "      ") << (torch ? "*" : " ")
                     << "/------------------------------\\"
                     << (!torch ? "*" : " ") << (!adam ? "(Adam)" : "      ") << (!bob ? "(Bob)" : "     ")
                     << (!clair ? "(Clair)" : "       ") << (!dave ? "(Dave)" : "      ") << endl;

        // Player input with validation
        char x, y;
        bool errorflag = false;
        // Correct name for 1st guy, A, B, C or D
        cout << "Input first guy, A for Adam, B for Bob, and so on:" << endl;
        cin >> x; cin.ignore(100, '\n'); // cin.ignore is to ignore the rest of characters if user input more than one character in one line
        while (!(x == 'A' || x == 'B' || x == 'C' || x == 'D')){
            cout << "Invalid input!" << endl;
            cout << "Name begins with '" << x << "' doesn't exist" << endl;
            cin >> x; cin.ignore(100, '\n');
        }

        // Correct 'name' for 2nd guy, A, B, C, D or N (for nobody)
        cout << "Input second guy A for Adam, B for Bob, ... If nobody, input N:" << endl;
        cin >> y; cin.ignore(100, '\n');
        while (!(y == 'A' || y == 'B' || y == 'C' || y == 'D' || y == 'N')){
            cout << "error: invalid input!" << endl;
            cout << "Please input N or the upper case first letter of a name:" << endl;
            cin >> y; cin.ignore(100, '\n');
        }

        // warning for two same person
        if (x == y)
        {
            cout << "Warning: the two people are the same, only one will count" << endl;
            y = 'N';
        }
        // the torch should be with them
        if ((torch^adam) && (x == 'A' || y == 'A')) errorflag = true; // adam is not on the torch side
        if ((torch^bob) && (x == 'B' || y == 'B')) errorflag = true; // the same
        if ((torch^clair) && (x == 'C' || y == 'C')) errorflag = true;
        if ((torch^dave) && (x == 'D' || y == 'D')) errorflag = true;
        if (errorflag) // if logical error occur, output some info and ask to input again!
        {
            cout << "Error: The torch is not with them!" << endl;
            continue;
        }

        // up to here all inputs are validated

        // cross the bridge
        if (y == 'N')
            cout << x << " is crossing the bridge with torch!" << endl;
        else
            cout << x << " and " << y << " are crossing the bridge with torch!" << endl;

        cout << "/-------------";
        if (y == 'N')
            cout << (torch ? "" : "--<<--") << x << (torch ? "*-->>" : "") << "-----------------\\" << endl;
        else
            cout << (torch ? "" : "--<<--") << x << " and " << y << (torch ? "*-->>" : "") << "------------\\" << endl;
        cout << '\n' << endl;

        switch(x)
        {
        case 'A': adam = !adam; break;
        case 'B': bob = !bob; break;
        case 'C': clair = !clair; break;
        case 'D': dave = !dave; break;
        }
        switch(y)
        {
        case 'A': adam = !adam; break;
        case 'B': bob = !bob; break;
        case 'C': clair = !clair; break;
        case 'D': dave = !dave; break;
        default: break; // may be n and N
        }
        torch = !torch; // don't forget to switch the torch side

        /***********************
         * calculate time passed
         * Note that the person that takes longer time determines the time
         *************************/
        int time;
        int time_x = 0, time_y = 0;
        switch(x)
        {
        case 'A': time_x = ADAM_TIME; break;
        case 'B': time_x = BOB_TIME; break;
        case 'C': time_x = CLAIR_TIME; break;
        case 'D': time_x = DAVE_TIME; break;
        }
        switch(y)
        {
        case 'A': time_y = ADAM_TIME; break;
        case 'B': time_y = BOB_TIME; break;
        case 'C': time_y = CLAIR_TIME; break;
        case 'D': time_y = DAVE_TIME; break;
        default: break; // in case that y == 'N', we remain time_y to be 0 so that time_x always win
        }
        // the person that takes longer time determines the final time
        time = (time_y > time_x ? time_y : time_x);

        cout << time << "min" << (time == 1 ? "" : "s") <<  " passed..." << endl;
        time_used += time;

        /*******************
         * Check if we already achieve the final goal
         * Also compared to optimal time and output congratulation if optimal solution is got
         ***********************/
        if (!adam && !bob && !clair && !dave) // the condition that every one has reach right side (all booleans are false)
        {
            cout << "                         /------------------------------\\*(Adam)(Bob)(Clair)(Dave)" << endl << endl;;
            cout << "They finally made it to the other side!" << endl;
            cout << "You totally take " << time_used << " mins" << endl;
            if (time_used > OPTIMAL_TIME)
                cout << "The optimal time is " << OPTIMAL_TIME << " mins, you can do better next time!" << endl;
            else
                cout << "Congratulation! You have got the optimal solution!" << endl;
            break;
        }
    }
    system("pause");
    return 0;
}
