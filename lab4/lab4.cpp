//
//  lab4.cpp
//  Copyright © 2020 TrebledJ. All rights reserved.
//

//
//  it is to my deepest regret that I used camel case
//

#include "lab_helpers.hpp"
#include "lab4_board.hpp"

#include <iostream>
#include <string>

using namespace std;


const char QUIT_INSTRUCTION = '0';
const int  Size3x3 = 3;
const int  Size4x4 = 4;

class Application
{
    Scanner sc;
    Printer print;
public:
    Application()
    {
        print.set_sep("");
    }
    
    int run()
    {
        print("Welcome to mini-2048! Enter 'h' for help.");
        print("Enjoy the game!"); print();
        
        
        Board game{Size3x3, 32};
        game.print();
        
        bool reached_goal = false;
        
        while (1) {
            char inst = getInstruction();
            if (inst == QUIT_INSTRUCTION)
                break;
            if (inst == 'n' || inst == 'N') {
                game.startNewGame();
                game.print();
                continue;
            }
            
            auto dir = charToDirection(inst);
            
            bool changed = game.move(dir);
            print();
            game.print();
            
            if (!changed)
                cout << "!! the board did not change !!" << endl;
            
            if (game.lost()) {
                print("!! no more moves left !!");
                cout << "(enter 'n' to start a new game)" << endl << endl;
                continue;
            }
            
            print();

            
            if (game.win() && !reached_goal) {
                reached_goal = true;
                
                print("You've reached ", game.goal, ". Congratulations!"); print();
                char cont = sc.get_a<char>("continue? (y/n) >>> ");
                cin >> cont; cin.ignore();
                if (cont == 'n')
                    break;
                
                print("Good luck!"); print();
                game.print();
            }
        }
        
        print();
        print("Good-bye! o/");
        return 0;
    }
    
    
private:
    char getInstruction() {
        char d = sc.get_a<char>(">>> ");
        switch (d) {
            case 'w': case 'W':
            case 'a': case 'A':
            case 's': case 'S':
            case 'd': case 'D':
            case 'n': case 'N':
            case QUIT_INSTRUCTION:
                return d;
            case 'h': case 'H':
                print("w: move tiles up");;
                print("a: move tiles left");
                print("s: move tiles down");
                print("d: move tiles right");
                print("n: start new game");
                print("0: quit game");
                print();
                return getInstruction();
            default:
                print("Enter 'h' for help"); print();
                return getInstruction();
        }
    }

    Direction charToDirection(char d) {
        switch (d) {
            case 'w': case 'W':
                return UP;
            case 'a': case 'A':
                return LEFT;
            case 's': case 'S':
                return DOWN;
            case 'd': case 'D':
                return RIGHT;
            default:
                print("!!! run-time error: detected non-directional character: '", d, "' !!!");
                return UP;
        }
    }
};


int main() {
    Application app;
    return app.run();
}
