//
//  lab7.cpp
//  Copyright © 2020 TrebledJ. All rights reserved.
//

#include "lab_helpers.hpp"
#include "lab7_board.hpp"

#include <iostream>


using namespace std;


int main()
{
    Scanner input;
    Printer print;
    
    auto size = input.get_a<unsigned>("(size) >>> ");
    auto use_solution = input.get_a<bool>("(use solution <1-Yes / 0-No>) >>> ");
    
    Board board{size, use_solution};
    
    while (1)
    {
        print();
        board.print();
        
        if (board.win())
        {
            print("Congratulations! You win!");
            break;
        }
        
        auto [x, y] = input.get<unsigned, unsigned>("(location <x, y>) >>> ");
        board.toggle(x, y);
    }
}
