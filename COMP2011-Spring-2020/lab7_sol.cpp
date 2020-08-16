#include <iostream>
#include <ctime>
#include <cstdlib>
#include <math.h>
using namespace std;

const int MAX_SIZE = 10;

void step(int dest[][MAX_SIZE], int x, int y, int size);
void init_figure(int dest[][MAX_SIZE], int size);
bool check_win(const int dest[][MAX_SIZE], int size);
void get_candidate(int dest[][MAX_SIZE], int size, int num);
void find_solution(const int dest[][MAX_SIZE], int size);

//Helper functions for printing
void printw(int size, int ch) { cout.width(size); cout << ch; }
void printw(int size, char ch) { cout.width(size); cout << ch; }
void print_figure(const int figure[][MAX_SIZE], int size) {
    const int gap = 3;
    cout.fill(' ');
    printw(gap, ' ');
    printw(gap, ' ');
    for (int i = 0; i < size; ++i) {
        printw(gap, i);
    }
    cout << endl;
    printw(gap, ' ');
    printw(gap, '+');
    for (int i = 0; i < size; ++i) printw(gap, '-');
    cout << endl;
    for (int i = 0; i < size; ++i) {
        printw(gap, i);
        printw(gap, '|');
        for (int j = 0; j < size; ++j) {
            if (figure[i][j] == 0)
                printw(gap, '.');
            else
                printw(gap, 'O');
        }
        cout << endl;
    }
}

// generate a puzzle.
void init_figure(int dest[][MAX_SIZE], int size) {
    // set random seed
    srand(static_cast<unsigned int>(time(0)));

    // The puzzle is generated by a reverse solving process which makes sure the it has the solution.
    // initialization the whole table with 0 values.
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j)
            dest[i][j] = 0;
    // randomly flip some lights as well as their neighbors.
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j)
            if (rand() % 2 == 1)
                step(dest, i, j, size);
}

// toggle one light as well as its immediate neighbors
// dest -- the 2D array that stores the states of all lights
// x, y -- the location (x, y) that is clicked in this step
// size -- the size of the grid
void step(int dest[][MAX_SIZE], int x, int y, int size) {
    dest[x][y] = 1 - dest[x][y];
    if (x > 0)
        dest[x - 1][y] = 1 - dest[x - 1][y];
    if (y > 0)
        dest[x][y - 1] = 1 - dest[x][y - 1];
    if (x < size - 1)
        dest[x + 1][y] = 1 - dest[x + 1][y];
    if (y < size - 1)
        dest[x][y + 1] = 1 - dest[x][y + 1];
}

// check if we win, return True for win and False otherwise.
// dest -- the 2D array that stores the states of all lights
// size -- the size of the grid
bool check_win(const int dest[][MAX_SIZE], int size) {
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j)
            if (dest[i][j] == 1) return false;
    return true;
}

// convert the number of the candidate into the solution.
void get_candidate(int dest[][MAX_SIZE], int size, int num) {
    int i = 0;
    // The number of the candidate is a decimal number, which is convertted into a binary number. Then each figure of the binary number is assigned into a 2D array.
    while(num > 0){
        dest[i / size][i % size] = num % 2;
        num /= 2;
        ++i;
    }
}

// search for the solution by enumeration.
// dest -- the 2D array that stores the states of all lights
// size -- the size of the grid
void find_solution(const int dest[][MAX_SIZE], int size) {
    int solution[MAX_SIZE][MAX_SIZE] = {};
    
    // we have to test candidates on a new table to avoid modification of the origin table.
    int new_figure[MAX_SIZE][MAX_SIZE] = {};
    int k = 0;

    for (; k < pow(2, size * size); ++k){
        get_candidate(solution, size, k);
        
        // copy the origin table into a new table.
        for (int x = 0; x < size; ++x)
            for (int y = 0; y < size; ++y)
                new_figure[x][y] = dest[x][y];
        
        for (int x = 0; x < size; ++x)
            for (int y = 0; y < size; ++y)
                if (solution[x][y] == 1)
                    step(new_figure, x, y, size);
        
        if (check_win(new_figure, size)) break;
    }
    
    if (k < pow(2, size * size)){
        // print the solution
        cout << endl << "Solution:" << endl;
        for (int x = 0; x < size; ++x)
            for (int y = 0; y < size; ++y)
                if (solution[x][y] == 1)
                    cout << "(" << x << "," << y << ")" << " ";
        cout << endl;
    }
    else
        cout << endl << "No solution!" << endl;
}

// main function
int main() {
    int size;
    cout << "Welcome to Lights Out Puzzle!" << endl << "Please input the size." << endl;
    cin >> size;
    
    int is_show_solution;
    cout << endl << "Do you want the solution? (1-Yes / 0-No)" << endl;
    cin >> is_show_solution;

    // figure is a 2D array that stores the states of all lights. 1 for on and 0 for off.
    int figure[MAX_SIZE][MAX_SIZE] = {};

    init_figure(figure, size);
    
    int x = -1, y = -1;
    do{
        print_figure(figure, size);
        if (check_win(figure, size)){
            cout << endl << "******Congratulations!******" << endl;
            return 0;
        }
        
        if (is_show_solution)
            find_solution(figure, size);
        cout << "Please choose a location (x, y)" << endl;
        cin >> x >> y;
        step(figure, x, y, size);
    }while(x >= 0 && y >= 0);

    return 0;
}