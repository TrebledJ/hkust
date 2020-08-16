#include <iostream>
using namespace std;

/*
 * (Given)
 * Print the current game board
 */
void print(const char board[], int valid_length)
{
    cout << " [";
    for (int i = 0; i < valid_length; ++i)
       cout << i;
    cout << "]" << endl;
    cout << "  ";
    for (int i = 0; i < valid_length; ++i)
       cout << board[i];
    cout << endl;
}

/*
 * Initialize the game board with white (W) marbles on the left and
 * black (B) marbles on the right, and a gap in between
 * Returns the length of the puzzle, i.e. num_W + 1 + num_B
 */
int initialize(char board[], int num_W, int num_B)
{
    // TODO
    int valid_length = num_W + num_B + 1;
    for (int i = 0; i < num_W; ++i)
    {
        board[i] = 'W';
    }
    board[num_W] = '.';
    for (int i = num_W + 1; i < valid_length; ++i)
    {
        board[i] = 'B';
    }
    return valid_length;
}

/*
 * Jump a marble over 1 and only 1 marble of the opposite color into the empty position.
 * You CANNOT jump marbles over more than 1 position, and
 * you CANNOT backtrack your moves (B can only be moved to left, and W can only be moved to right).
 *
 * Returns true if the jump is valid
 * otherwise, returns false
 */
bool jump(char board[], int length, int index)
{
    // TODO
    // index out of range or position not empty
    if (index >= length || index < 0 || board[index] == '.')
    {
        return false;
    }
  
    // move leftwards if the selected marble is black (B)
    if (board[index] == 'B' && index - 2 >= 0 && board[index - 2] == '.' && board[index - 1] == 'W')
    {
        board[index - 2] = board[index];
        board[index] = '.';
        return true;
    }
    // move rightwards if the selected marble is white (W)
    else if (board[index] == 'W' && index + 2 < length && board[index + 2] == '.' && board[index + 1] == 'B')
    {
        board[index + 2] = board[index];
        board[index] = '.';
        return true;
    }
    return false;
}

/*
 * Slide a marble 1 space (into the empty position)
 * you CANNOT backtrack your moves (B can only be moved to left, and W can only be moved to right).
 *
 * Returns true if the slide is valid
 * otherwise, returns false
*/
bool slide(char board[], int length, int index)
{
    // TODO
    // index out of range or position not empty
    if (index >= length || index < 0 || board[index] == '.')
    {
        return false;
    }
    bool is_left = (board[index] == 'B');
    // move leftwards if the selected marble is black (B)
    if (is_left && index - 1 >= 0 && board[index - 1] == '.')
    {
        board[index - 1] = board[index];
        board[index] = '.';
        return true;
    }
    // move rightwards if the selected marble is white (W)
    else if (!is_left && index + 1 < length && board[index + 1] == '.')
    {
        board[index + 1] = board[index];
        board[index] = '.';
        return true;
    }
    return false;
}

/*
 * Returns true if all black marbles are on the left and white marbles are on the right
 * otherwise, returns false
 */
bool game_finished(const char board[], int num_W, int num_B)
{
    // TODO
    if (board[num_B] == '.')
    {
        for (int i = 0; i < num_B; ++i)
        {
            if (board[i] != 'B')
            {
                return false;
            }
        }
        return true;
    }
    return false;
}

int main()
{
    char board[1000] = {};
    int num_W, num_B;

    // Get the number of white (W) & black (B) marbles
    cout << "Num of white and black marbles: ";
    cin >> num_W >> num_B;

    // Initialize the board
    int length = initialize(board, num_W, num_B);
    print(board, length);

    // Continue while not all marbles are switched
    while(!game_finished(board, num_W, num_B))
    {
        // Get the index (position) for the move (operation), -1 means give up the game
        int index;
        cout << "Index (-1 to exit): ";
        cin >> index;
        if(index == -1)
        {
            cout << "Exit." << endl;
            break;
        }

        // Get the operation, 'J' for jump or 'S' for slide
        char op;
        cout << "'J' or 'S': ";
        cin >> op;
        bool res = false;
        switch (op)
        {
        case 'J':
            res = jump(board, length, index);
            break;
        case 'S':
            res = slide(board, length, index);
            break;
        }
        if(!res)
            cout << "Error!" << endl;
        else
            print(board, length);
    }

    if(game_finished(board, num_W, num_B))
    {
        cout << "Congratulations!" << endl;
    }

    return 0;
}
