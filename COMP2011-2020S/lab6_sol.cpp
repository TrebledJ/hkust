#include <iostream>
#include <ctime>
#include <cstdlib>

using namespace std;

// size of figure, it should be 15
const int SIZE = 15;
// whether print debug information
bool DEBUG = false;

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

    // base cases
    if (x < 0 || x >= SIZE || y < 0 || y >= SIZE)
        return 0;
    if (figure[x][y] == 'C'){
        if (debug) cout << "(" << x << ", " << y << ") is a cat" << endl;
        return 0;
    }
    if (figure[x][y] == 'Z')
        return 0;

    // bite (x, y)
    figure[x][y] = 'Z';
    if (debug) cout << "(" << x << ", " << y << ") changes to Zombie" << endl;
    int count = 1;

    // bite neighbours recursively at 4 directions
    count += bite(figure, x + 1, y, is_lazy, debug);
    count += bite(figure, x - 1, y, is_lazy, debug);
    count += bite(figure, x, y + 1, is_lazy, debug);
    count += bite(figure, x, y - 1, is_lazy, debug);
    if (!is_lazy) { // bite 4 more directions if the zombie is scary
        count += bite(figure, x-1, y-1, is_lazy, debug);
        count += bite(figure, x-1, y+1, is_lazy, debug);
        count += bite(figure, x+1, y-1, is_lazy, debug);
        count += bite(figure, x+1, y+1, is_lazy, debug);
    }
    return count;
}

// helper functions for random_figure()
// use integer (x, y) in ellipse
int square(int num) { return num * num; }
int ellipse(int x, int y, int center_x, int center_y, int a, int b) {
    return square(b*(x - center_x)) + square(a*(y - center_y)) - square(a*b);
}
//Put 'ch' not only on (x, y), but also on three symmetric location.
void symmetry_put(char figure[][SIZE], int center_x, int center_y,
        int x, int y, char ch) {
    int sym_x = 2*center_x - x;
    int sym_y = 2*center_y - y;
    if (x >= 0 && x < SIZE && y >= 0 && y < SIZE) figure[x][y] = 'C';
    if (x >= 0 && x < SIZE && sym_y >= 0 && sym_y < SIZE) figure[x][sym_y] = 'C';
    if (sym_x >= 0 && sym_x < SIZE && y >= 0 && y < SIZE) figure[2*center_x-x][y] = 'C';
    if (sym_x >= 0 && sym_x < SIZE && sym_y >= 0 && sym_y < SIZE) figure[2*center_x-x][2*center_y-y] = 'C';
}

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
    for (int i = 0; i < SIZE; ++i)
        for (int j = 0; j < SIZE; ++j)
            figure[i][j] = '.';
    int x0 = (rand() % (SIZE / 2) + SIZE / 4);
    int y0 = (rand() % (SIZE / 2) + SIZE / 4);
    int a = ((rand() % (SIZE / 4)) + SIZE / 4);
    int b = ((rand() % (SIZE / 4)) + SIZE / 4);
    int y = y0 + b;
    int draw = y;
    for (int x = x0 ; x <= x0 + a; ++x) {
        int current = ellipse(x, y, x0, y0, a, b);
        int next = ellipse(x, y - 1, x0, y0, a, b);
        while (current*next > 0) {
            y--;
            current = ellipse(x, y, x0, y0, a, b);
            next = ellipse(x, y - 1, x0, y0, a, b);
        }
        int last_draw = draw;
        if (abs(current) <= abs(next)) {
            draw = y;
        } else {
            draw = y - 1;
        }
        int diff = last_draw - draw;
        for (int i = 0; i <= diff / 2; ++i)
            symmetry_put(figure, x0, y0, x, draw+i, 'C');
        for (int i = 0; i < diff / 2; ++i) {
            symmetry_put(figure, x0, y0, x, draw+i, 'C');
            if (figure[x][last_draw-1-i] != 'C')
                symmetry_put(figure, x0, y0, x-1, last_draw-1-i, 'C');
        }
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
