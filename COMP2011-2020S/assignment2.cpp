/*
 * COMP2011 (Spring 2020) Assignment 2: "Recursion"
 *
 * Student name: [redacted]
 * Student ID: [redacted]
 * Student email: [redacted]
 *
 * Note:
 * - DO NOT change any of the function headers given
 * - DO NOT use any loops
 * - DO NOT use any global variables or add additional libraries.
 * - You can add helper function(s) if needed.
 */

#include <iostream>

using namespace std;

// Constants

// NULL character. This is the last char of all C Strings
const char END = '\0';

// Single quotation character '
const char SQUOTE = '\'';

// Double quotation character "
const char DQUOTE = '\"';

// Error. Used in Task 2 and 3
const int ERROR = -1;

// Practice Task: Task 0 (Not Graded)
unsigned int recursive_strlen(const char line[], int start)
{
    return line[start] == END ? 0 : (recursive_strlen(line, start+1) + 1);
}


namespace Helper
{

    int find_first_of(const char line[], int start, char c);
    unsigned int count_consecutive_dquotes(const char line[], int i);
    template<class T> T const& max(T const& a, T const& b);

}


// Normal Task: Task 1
unsigned int count_dquotes(const char line[], int start)
{
    return line[start] == END ? 0 : (count_dquotes(line, start+1) + (line[start] == DQUOTE));
}


// Normal Task: Task 2
int find_first_dquote(const char line[], int start)
{
    return Helper::find_first_of(line, start, DQUOTE);
}


// Normal Task: Task 3
int count_chars_in_matched_dquote(const char line[], int start)
{
    switch (line[start])
    {
        case END:
            return 0;
        case DQUOTE:
        {
            //  skip to next dquote
            int next = find_first_dquote(line, start+1);
            
            //  no match
            if (next == ERROR)
                return ERROR;
            
            //  count dquotes from positional difference
            int rest = count_chars_in_matched_dquote(line, next+1);
            return rest == ERROR ? ERROR : (rest + (next - start - 1));
        }
        default:
        {
            //  skip to next dquote
            int next = find_first_dquote(line, start+1);
            return next == ERROR ? 0 : count_chars_in_matched_dquote(line, next);
        }
    }
}


// Challenging Task: Task 4
bool check_quotes_matched(const char line[], int start)
{
    switch (line[start])
    {
        case END:
            return true;
        case SQUOTE:
        case DQUOTE:
        {
            int next = Helper::find_first_of(line, start+1, line[start]);
            return next == ERROR ? false : check_quotes_matched(line, next+1);
        }
        default:
            return check_quotes_matched(line, start+1);
    }
    
    return line;
}


// Challenging Task: Task 5
unsigned int length_of_longest_consecutive_dquotes(const char line[], int start)
{
    unsigned int n = Helper::count_consecutive_dquotes(line, start);
    return line[start] == END ? 0 : Helper::max(n, length_of_longest_consecutive_dquotes(line, start+n+1));
}


//  helper functions
namespace Helper
{

    int find_first_of(const char line[], int start, char c)
    {
        if (line[start] == END)
            return ERROR;
        return line[start] == c ? start : find_first_of(line, start+1, c);
    }

    unsigned int count_consecutive_dquotes(const char line[], int i) {
        return line[i] != DQUOTE ? 0 : count_consecutive_dquotes(line, i+1) + 1;
    }

    template<class T>
    T const& max(T const& a, T const& b) {
        return a > b ? a : b;
    }

}


// DO NOT WRITE ANYTHING AFTER THIS LINE. ANYTHING AFTER THIS LINE WILL BE REPLACED

#include "assignment2_assertions.hpp"
const int MAX_LENGTH = 1000;

int main()
{
    int option = 0;
    char line[MAX_LENGTH];
    Assertions::test();
    do {
        cout << "Options:" << endl;
        cout << "0:  Test recursive_strlen()" << endl;
        cout << "1:  Test count_dquotes()" << endl;
        cout << "2:  Test find_first_dquote()" << endl;
        cout << "3:  Test count_chars_in_matched_dquote()" << endl;
        cout << "4:  Test check_quotes_matched()" << endl;
        cout << "5:  Test length_of_longest_consecutive_dquotes()" << endl;
        cout << "-1: Quit" << endl;

        cin >> option;
        cin.ignore();

        switch (option) {
            case 0:
            cout << "Testing recursive_strlen()" << endl;
            cout << "Enter line: ";
            cin.getline(line, MAX_LENGTH);
            cout << recursive_strlen(line, 0) << endl;
            break;

            case 1:
            cout << "Testing count_dquotes()" << endl;
            cout << "Enter line: ";
            cin.getline(line, MAX_LENGTH);
            cout << count_dquotes(line, 0) << endl;
            break;

            case 2:
            cout << "Testing find_first_dquote()" << endl;
            cout << "Enter line: ";
            cin.getline(line, MAX_LENGTH);
            cout << find_first_dquote(line, 0) << endl;
            break;

            case 3:
            cout << "Testing count_chars_in_matched_dquote()" << endl;
            cout << "Enter line: ";
            cin.getline(line, MAX_LENGTH);
            cout << count_chars_in_matched_dquote(line, 0) << endl;
            break;

            case 4:
            cout << "Testing check_quotes_matched()" << endl;
            cout << "Enter line: ";
            cin.getline(line, MAX_LENGTH);
            cout << check_quotes_matched(line, 0) << endl;
            break;

            case 5:
            cout << "Testing length_of_longest_consecutive_dquotes()" << endl;
            cout << "Enter line: ";
            cin.getline(line, MAX_LENGTH);
            cout << length_of_longest_consecutive_dquotes(line, 0) << endl;
            break;

            default:
            break;
        }

        cout << endl;

    } while (option != -1);

    return 0;
}
