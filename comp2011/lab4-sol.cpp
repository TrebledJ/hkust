#include <iostream>
#include <stdlib.h>
#include <time.h>

#define WIN_NUM 32

using namespace std;

//Tiles slide as far as possible in the chosen direction until they are stopped by either another tile or the edge of the grid
//If two tiles of the same number collide while moving, they will merge into a tile with the total value of the two tiles that collided.
//The resulting tile cannot merge with another tile again in the same move.

void slide(int &n1, int &n2, int &n3)
{
    // see if anything can merge
    // if some element can be merged, merged it to one element and the other set to 0
    if (n2 == n3)
    {
        n2 += n3;
        n3 = 0;
    }
    else if (n1 == n2)
    {
        n1 += n2;
        n2 = 0;
    }
    else if (n1 == n3 && n2 == 0) // this is a special case
    {
        n3 += n1;
        n1 = 0;
    }
    // slide to right
    // after slide, all non-zero elements will be on the right
    if (n3 == 0 && n2 != 0)
    {
        n3 = n2;
        n2 = n1;
        n1 = 0;
    }
    else if (n3 == 0 && n2 == 0)
    {
        n3 = n1;
        n1 = 0;
    }
    else if (n3 != 0 && n2 == 0)
    {
        n2 = n1;
        n1 = 0;
    }
    return;
}

void putTile(int &n1, int &n2, int &n3, int &n4, int &n5, int &n6, int &n7, int &n8, int &n9)
{
    // generate 2 or 4 randomly
    int num = (rand() % 2 + 1) * 2;

    bool put = false;
    do {
        int position = rand() % 9;
        // randomly generate a position, if the position is not occupied, set that to num
        switch (position)
        {
        case 0:
            if (n1 == 0) {n1 = num; put = true;} break;
        case 1:
            if (n2 == 0) {n2 = num; put = true;} break;
        case 2:
            if (n3 == 0) {n3 = num; put = true;} break;
        case 3:
            if (n4 == 0) {n4 = num; put = true;} break;
        case 4:
            if (n5 == 0) {n5 = num; put = true;} break;
        case 5:
            if (n6 == 0) {n6 = num; put = true;} break;
        case 6:
            if (n7 == 0) {n7 = num; put = true;} break;
        case 7:
            if (n8 == 0) {n8 = num; put = true;} break;
        case 8:
            if (n9 == 0) {n9 = num; put = true;} break;
        }
    }while (!put);
}

int max3(int a, int b, int c) {return a > b && a > c ? a : (b > c ? b : c);}
int min3(int a, int b, int c) {return a < b && a < c ? a : (b < c ? b : c);}

// user wins the game if there is 32 reached
// i.e. max is 32
bool checkWin(int n1, int n2, int n3, int n4, int n5, int n6, int n7, int n8, int n9)
{
    int a = max3(n1, n2, n3);
    int b = max3(n4, n5, n6);
    int c = max3(n7, n8, n9);
    return (WIN_NUM == max3(a, b, c));
}

// user loses the game if all grids are filled
// i.e. min > 0
bool checkLose(int n1, int n2, int n3, int n4, int n5, int n6, int n7, int n8, int n9)
{
    int a = min3(n1, n2, n3);
    int b = min3(n4, n5, n6);
    int c = min3(n7, n8, n9);
    return (min3(a, b, c) > 0);
}

// slide tiles according to given direction
void slideTiles(char direction, int& n1, int& n2, int& n3, int& n4, int& n5, int& n6, int& n7, int& n8, int& n9)
{
    switch (direction)
    {
    case 'w':
        slide(n7, n4, n1);
        slide(n8, n5, n2);
        slide(n9, n6, n3);
        break;
    case 's':
        slide(n1, n4, n7);
        slide(n2, n5, n8);
        slide(n3, n6, n9);
        break;
    case 'a':
        slide(n3, n2, n1);
        slide(n6, n5, n4);
        slide(n9, n8, n7);
        break;
    case 'd':
        slide(n1, n2, n3);
        slide(n4, n5, n6);
        slide(n7, n8, n9);
    }
}

// simplified 2048 game: 3x3 game board to reach 32
int main()
{
    // 3x3 empty game board
    int a11 = 0, a12 = 0, a13 = 0, a21 = 0, a22 = 0, a23 = 0, a31 = 0, a32 = 0, a33 = 0;

    cout << "Welcome to 2048 Game!" << endl
            << "input w|a|s|d to slide the tiles." << endl
            << "Game start! Get " << WIN_NUM <<" to win!" << endl << endl;

    while (true)
    {
        srand((unsigned)time(NULL));
        //new tile will randomly appear in an empty spot on the board with a value of either 2 or 4
        putTile(a11, a12, a13, a21, a22, a23, a31, a32, a33);

        //print game board
        cout << a11 << '\t' << a12 << '\t' << a13 << '\n'
                << a21 << '\t' << a22 << '\t' << a23 << '\n'
                << a31 << '\t' << a32 << '\t' << a33 << '\n'
                << endl;

        //grab player's choice
        cout << "Input direction:" << endl;
        char direction;
        cin >> direction;

        // slide tiles
        slideTiles(direction, a11, a12, a13, a21, a22, a23, a31, a32, a33);

        // check win
        if (checkWin(a11, a12, a13, a21, a22, a23, a31, a32, a33))
        {
            cout << "Congratulations! You Win!\n" << endl;
            break;
        }
        // check lose
        if (checkLose(a11, a12, a13, a21, a22, a23, a31, a32, a33))
        {
            cout << "Sorry! You Lose!\n" << endl;
            break;
        }
    } // end of game loop

    // final game board
    cout << a11 << '\t' << a12 << '\t' << a13 << '\n'
            << a21 << '\t' << a22 << '\t' << a23 << '\n'
            << a31 << '\t' << a32 << '\t' << a33 << '\n'
            << endl;
    return 0;
}
