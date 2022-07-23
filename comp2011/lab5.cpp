//
//  lab5.cpp
//  Copyright © 2020 TrebledJ. All rights reserved.
//

#include "lab_helpers.hpp"

#include <iostream>
#include <iomanip>

using namespace std;


const bool INTERACTIVE = true;


class Shuttle {
public:
    unsigned white;
    unsigned black;
    unsigned size;
    
private:
    static const char WHITE = 'W';
    static const char BLACK = 'B';
    static const char EMPTY = '.';
    static const char GHOST = ' ';
    char* shuttle;
    
public:
    Shuttle(unsigned white, unsigned black)
        : white{white}, black{black}, size{white + black + 1}, shuttle{new char[size]}
    {
        for (unsigned i = 0; i < white; ++i)
            shuttle[i] = WHITE;
        shuttle[white] = EMPTY;
        for (unsigned i = white + 1; i < size; ++i)
            shuttle[i] = BLACK;
    }
    ~Shuttle() { delete[] shuttle; }
    
    
    bool in_range(int index) const {
        return 0 <= index && index < size;
    }
    
    bool jump(unsigned index) {
        char c = at(index);
        if (c == EMPTY
            || (c == WHITE && (at(index + 1) != BLACK || at(index + 2) != EMPTY))
            || (c == BLACK && (at(index - 1) != WHITE || at(index - 2) != EMPTY)) ) {
            return false;
        }
        
        if (c == WHITE) {
            shuttle[index + 2] = c;
            shuttle[index] = EMPTY;
        }
        if (c == BLACK) {
            shuttle[index - 2] = c;
            shuttle[index] = EMPTY;
        }
        
        return true;
    }
    
    bool slide(unsigned index) {
        char c = at(index);
        if (c == EMPTY
            || (c == WHITE && at(index + 1) != EMPTY)
            || (c == BLACK && at(index - 1) != EMPTY) ) {
            return false;
        }
        
        if (c == WHITE) {
            shuttle[index + 1] = c;
            shuttle[index] = EMPTY;
        }
        if (c == BLACK) {
            shuttle[index - 1] = c;
            shuttle[index] = EMPTY;
        }

        return true;
    }
    
    void print() const {
        for (unsigned i = 0; i < size; ++i)
            cout << setw(4) << i;
        cout << endl;
        for (unsigned i = 0; i < size; ++i)
            cout << setw(4) << shuttle[i];
        cout << endl;
    }
    
    bool win() const {
        for (unsigned i = 0; i < black; ++i)
            if (shuttle[i] != BLACK)
                return false;
        
        if (shuttle[black] != EMPTY) return false;
        
        for (unsigned i = black + 1; i < size; ++i)
            if (shuttle[i] != WHITE)
                return false;
        
        return true;
    }

private:
    char at(int index) const {
        if (!in_range(index))
            return GHOST;
        
        return shuttle[index];
    }
    
};


int main() {
    Scanner sc;
    sc.set_interactive(INTERACTIVE);
    Printer print{std::cout, ""};
    
    auto [white, black] = sc.get<unsigned, unsigned>("(<white> <black>) >>> ");
    Shuttle shuttle{white, black};
    
    while (1) {
        if (INTERACTIVE) {
            print();
            shuttle.print();
        }
        
        auto index = sc.get_a<int>("(<index>) >>> ", false);
        
        if (index == -1) {
            print("Exit.");
            break;
        }
        if (!shuttle.in_range(index)) {
            print("index(", index, ") not in range([0, ", shuttle.size-1, "])");
            continue;
        }
        
        auto cmd = sc.get_a<char>("(<J/S>) >>> ");
        if (INTERACTIVE) print();
        
        bool ok = true;
        switch (cmd) {
            case 'j': case 'J': ok = shuttle.jump(index); break;
            case 's': case 'S': ok = shuttle.slide(index); break;
            default:
                print("command '", cmd, "' not recognised");
                continue;
        }
        if (!ok) {
            print("Error!");
            continue;
        }
        if (shuttle.win()) {
            print("Congratulations!");
            break;
        }
    }
}
