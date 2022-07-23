//
//  lab6.cpp
//  Copyright © 2020 TrebledJ. All rights reserved.
//

//  so I used the skeleton because I was lazy ._.

#include <iostream>
#include <string>
#include <ctime>
#include <cstdlib>

using namespace std;

#undef DEBUG

// size of figure
const int SIZE = 15;
// whether print debug information
bool DEBUG = true;


namespace Symbol {
const char HUMAN = '.';
const char CAT = 'C';
const char ZOMBIE = 'Z';
}


//Helper functions for printing
void printw(int size, int ch) { cout.width(size); cout << ch; }
void printw(int size, char ch) { cout.width(size); cout << ch; }
void print_figure(const char figure[][SIZE]) {
    const int gap = 3;
    cout.fill(' ');
    printw(gap, ' ');
    printw(gap, ' ');
    for (int i = 0; i < SIZE; ++i) {
        printw(gap, i);
    }
    cout << endl;
    printw(gap, ' ');
    printw(gap, '+');
    for (int i = 0; i < SIZE+2; ++i) printw(gap, '-');
    cout << ">y\n";
    for (int i = 0; i < SIZE; ++i) {
        printw(gap, i);
        printw(gap, '|');
        for (int j = 0; j < SIZE; ++j) {
            printw(gap, figure[i][j]);
        }
        cout << endl;
    }
    for (int i = 0; i < 2; ++i) {
        printw(gap, ' ');
        printw(gap, '|');
        cout << endl;
    }
    printw(gap, ' ');
    printw(gap, 'v');
    cout << endl;
    printw(gap, ' ');
    printw(gap, 'x');
    cout << endl;
}

/**
 * Helper function for copying 'src' to 'dest'.
 */
void copy_figure(char dest[][SIZE], const char src[][SIZE]) {
    for (int i = 0; i < SIZE; ++i)
        for (int j = 0; j < SIZE; ++j)
            dest[i][j] = src[i][j];
}

/**
 * The human in (x, y) is bitten, and turn into a zombie.
 * It then bites four or eight of its neighbours and changes them into zombies.
 * New zombies bite their neighbours
 * It goes on and on...until
 * out of boundary, or the neighbour is a cat!
 * is_lazy:
 *     true, zombie bites 4 neighbours, up/down/left/right; false, zombie bites all 8 neighbours
 * debug:
 *     true, print debug information, false otherwise
 * Return:
 *     The number of times 'bite' is called.
 */
int bite(char figure[][SIZE], int x, int y, bool is_lazy, bool debug) {
    int nb_bites = 0;
    if (char c = figure[x][y]; c == Symbol::CAT) {
        if (debug) cout << "(" << x << ", " << y << ") is a cat" << endl;
        return 0;
    } else if (c != Symbol::HUMAN) {
        return 0;
    }
    
    if (debug) cout << "(" << x << ", " << y << ") is bitten" << endl;
    figure[x][y] = Symbol::ZOMBIE;
    
    
    const int dx[] = {1, 0, -1, 0, 1, 1, -1, -1};
    const int dy[] = {0, 1, 0, -1, 1, -1, 1, -1};
    const int z_bites = (is_lazy ? 4 : 8);
    for (int i = 0; i < z_bites; ++i) {
        int x2 = x + dx[i], y2 = y + dy[i];
        if (0 <= x2 && x2 < SIZE && 0 <= y2 && y2 < SIZE) {
            if (figure[x2][y2] == Symbol::HUMAN)
                nb_bites += bite(figure, x2, y2, is_lazy, debug);
        }
    }
    
    return nb_bites + 1;
}


struct Ellipse
{
    int x0, y0;
    unsigned a, b;
    Ellipse() : Ellipse(0, 0, 0, 0) {}
    Ellipse(int x0, int y0, unsigned a, unsigned b) : x0{x0}, y0{y0}, a{a}, b{b} {}
    bool contains(int x, int y) const
    {
        return (x - x0)*(x - x0)*b*b + (y - y0)*(y - y0)*a*a <= a*a*b*b;
    }
};
/**
 *
 * Create a random elliptical cat kingdom using the standard form
 * (x - x0)^2/a^2 + (y - y0)^2/b^2 == 1,
 * where (x0, y0) is the center of the ellipse.
 *
 * center point (x0, y0) is randomly located in the region [SIZE/4, 3*SIZE/4]x[SIZE/4, 3*SIZE/4],
 * and a, b in interval [SIZE/4, SIZE/2] so that in most cases the complete
 * curves can lies in figure.
 *
 * Hint: the basic principle is the same as what you did in lab3.
 *       Use the equation to test which points are inside or outside of the curve.
 *       Your task is to find the closest points to the curve.
 *       For example, you may start from the center (x0, y0), keep moving up until
 *       you find a point (x0, y1), such as
 *             (x0 - x0)^2/a^2 + (y1 - y0)^2/b^2 >= 1,
 *             (x0 - x0)^2/a^2 + ((y1 - 1)- y0)^2/b^2 <= 1,
 *       Then (x0, y1 - 1) and (x0, y1) are candidates which are points closest
 *       to the curve on the line x = x0.
 *       Choosing whichever one depends on your implementation.
 *       As in lab3, you can just try to draw a quarter of the ellipse, then copy this
 *       portion to three other symmetric positions
 */
void random_figure(char figure[][SIZE]) {
    //  dumb ellipse implementation
    //  I myself am not impressed
    //  the TA solution has a much better implementation (obviously)
    int x0 = 1+(SIZE/4) + rand() % (SIZE/2);
    int y0 = 1+(SIZE/4) + rand() % (SIZE/2);
    unsigned a = (SIZE/4)-1 + rand() % (SIZE/4);
    unsigned b = (SIZE/4)-1 + rand() % (SIZE/4);
    Ellipse e{x0, y0, a, b};
    
    int x = x0 + a;
    int y = y0;
    //  draw q1
    while (x != x0)
    {
        figure[x][y] = Symbol::CAT;
        x--;
        while (e.contains(x, ++y));
    }
    
    //  q2
    while (y != y0)
    {
        figure[x][y] = Symbol::CAT;
        y--;
        while (e.contains(--x, y));
    }
    
    //  q3
    while (x != x0)
    {
        figure[x][y] = Symbol::CAT;
        x++;
        while (e.contains(x, --y));
    }
    
    //  q4
    while (y != y0)
    {
        figure[x][y] = Symbol::CAT;
        y++;
        while (e.contains(++x, y));
    }
}

// Driver program
int main() {
    // set random seed
    srand(time(nullptr));

    // three 'cat kingdom' testing cases
    const char TESTING_CASES[][SIZE][SIZE] = { //compiler can automatically detect size of first dimension
            {
                    { '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', },
                    { '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', },
                    { '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', },
                    { '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', },
                    { '.', '.', '.', '.', '.', 'C', 'C', 'C', 'C', 'C', 'C', '.', '.', '.', '.', },
                    { '.', '.', '.', '.', 'C', '.', '.', '.', '.', '.', 'C', '.', '.', '.', '.', },
                    { '.', '.', '.', '.', 'C', '.', '.', '.', '.', '.', 'C', '.', '.', '.', '.', },
                    { '.', '.', '.', 'C', '.', '.', '.', '.', '.', 'C', 'C', '.', '.', '.', '.', },
                    { '.', '.', '.', '.', 'C', '.', '.', '.', '.', 'C', '.', '.', '.', '.', '.', },
                    { '.', '.', '.', '.', '.', 'C', 'C', 'C', 'C', 'C', '.', '.', '.', '.', '.', },
                    { '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', },
                    { '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', },
                    { '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', },
                    { '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', },
                    { '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', },
            },

            {
                    { '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', },
                    { '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', },
                    { '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', },
                    { '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', },
                    { '.', '.', '.', '.', '.', 'C', 'C', 'C', '.', 'C', 'C', '.', '.', '.', '.', },
                    { '.', '.', '.', '.', 'C', '.', '.', '.', '.', '.', 'C', '.', '.', '.', '.', },
                    { '.', '.', '.', '.', 'C', '.', '.', '.', '.', '.', 'C', '.', '.', '.', '.', },
                    { '.', '.', '.', 'C', '.', '.', '.', '.', '.', 'C', 'C', '.', '.', '.', '.', },
                    { '.', '.', '.', '.', 'C', '.', '.', '.', '.', 'C', '.', '.', '.', '.', '.', },
                    { '.', '.', '.', '.', '.', 'C', 'C', 'C', 'C', 'C', '.', '.', '.', '.', '.', },
                    { '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', },
                    { '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', },
                    { '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', },
                    { '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', },
                    { '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', },
            },

            {
                    { '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', },
                    { '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', },
                    { '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', },
                    { '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', },
                    { '.', '.', '.', '.', 'C', 'C', 'C', 'C', 'C', 'C', 'C', '.', '.', '.', '.', },
                    { '.', '.', '.', '.', 'C', '.', '.', '.', '.', '.', 'C', '.', '.', '.', '.', },
                    { '.', '.', '.', 'C', 'C', '.', '.', '.', '.', '.', 'C', '.', '.', '.', '.', },
                    { '.', '.', '.', 'C', '.', '.', '.', '.', '.', 'C', 'C', '.', '.', '.', '.', },
                    { '.', '.', '.', 'C', 'C', '.', '.', '.', '.', 'C', '.', '.', '.', '.', '.', },
                    { '.', '.', '.', '.', 'C', 'C', 'C', 'C', 'C', 'C', '.', '.', '.', '.', '.', },
                    { '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', },
                    { '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', },
                    { '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', },
                    { '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', },
                    { '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', },
            },
    };

    cout << "Welcome to ZOMBIE land!" << endl;
    cout << "Humans can only be protected by closed ring of cats!" << endl << endl;

    char figure[SIZE][SIZE] = {}; // working matrix

    int mode = 0;
    cout << "Which testing case do you prefer? (1/2/3) or generate random (4): ";
    cin >> mode; //assume input is always valid
    switch (mode) {
        case 1: copy_figure(figure, TESTING_CASES[0]); break;
        case 2: copy_figure(figure, TESTING_CASES[1]); break;
        case 3: copy_figure(figure, TESTING_CASES[2]); break;
//        default: cout << "Not Yet Implemented" << endl; return 0;
        // replace default case with the following one if you finish 'random_figure'
        default: random_figure(figure); break;
    }
    print_figure(figure);

    cout << "Please choose the location (x, y) of the first unlucky human: ";
    int x = 0, y = 0;
    cin >> x >> y;

    cout << "Lazy zombie bites 4 directions, scary zombie bites 8 directions!!" << endl;
    cout << "Is zombie lazy? (y/n): ";
    char is_lazy = 'y';
    cin >> is_lazy;

    cout << endl << "It's getting dark, zombie starts to bite!!!!" << endl;
    int steps = bite(figure, x, y, is_lazy == 'y', DEBUG);

    cout << endl << "After a scary night, the world becomes:" << endl << endl;
    print_figure(figure);
    cout << steps << " humans change to zombie!" << endl;
    return 0;
}
