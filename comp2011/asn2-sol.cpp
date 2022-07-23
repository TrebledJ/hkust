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


unsigned int recursive_strlen(const char line[], int start)
{
    char c = line[start];

    if (c == END)
        return 0;
    else
        return 1 + recursive_strlen(line, start + 1);
}


unsigned int count_dquotes(const char line[], int start)
{
    char c = line[start];

    if (c == END)
        return 0;
    else if (c == DQUOTE)
        return 1 + count_dquotes(line, start + 1);
    else
        return count_dquotes(line, start + 1);
}


int find_first_dquote(const char line[], int start)
{
    char c = line[start];
    if (c == END)
        return ERROR;
    else if (c == DQUOTE)
        return 0;
    else {
        int result = find_first_dquote(line, start + 1);
        if (result == ERROR)
            return ERROR;
        else
            return 1 + result;
    }
}


int task3_helper(const char line[], int start);

int count_chars_in_matched_dquote(const char line[], int start)
{
    char c = line[start];
    if (c == END)
        return 0;
    else if (c == DQUOTE) {
        return task3_helper(line, start + 1);
    }
    else
        return count_chars_in_matched_dquote(line, start + 1);
}

int task3_helper(const char line[], int start)
{
    char c = line[start];
    if (c == END)
        return ERROR;
    else if (c == DQUOTE)
        return count_chars_in_matched_dquote(line, start + 1);
    else {
        int result = task3_helper(line, start + 1);
        if (result == ERROR)
            return ERROR;
        else
            return 1 + result;
    }
}


bool check_quotes_matched_in_squote(const char line[], int start);
bool check_quotes_matched_in_dquote(const char line[], int start);

bool check_quotes_matched(const char line[], int start)
{
    char c = line[start];

    if (c == END)
        return true;
    else if (c == SQUOTE)
        return check_quotes_matched_in_squote(line, start + 1);
    else if (c == DQUOTE)
        return check_quotes_matched_in_dquote(line, start + 1);
    else
        return check_quotes_matched(line, start + 1);
}

bool check_quotes_matched_in_squote(const char line[], int start)
{
    char c = line[start];

    if (c == END)
        return false;
    else if (c == SQUOTE)
        return check_quotes_matched(line, start + 1);
    else
        return check_quotes_matched_in_squote(line, start + 1);
}

bool check_quotes_matched_in_dquote(const char line[], int start)
{
    char c = line[start];

    if (c == END)
        return false;
    else if (c == DQUOTE)
        return check_quotes_matched(line, start + 1);
    else
        return check_quotes_matched_in_dquote(line, start + 1);
}


unsigned int task5_helper(const char line[], int start, char prev, unsigned int counter);
unsigned int max(unsigned int a, unsigned int b);

unsigned int length_of_longest_consecutive_dquotes(const char line[], int start)
{
    if (line[start] == END)
        return 0;

    return task5_helper(line, start, 0, 0);
}

unsigned int task5_helper(const char line[], int start, char prev, unsigned int counter)
{
    char curr = line[start];

    if (curr == END)
        return 0;

    else if (curr == DQUOTE && prev == DQUOTE) {
        unsigned int result = task5_helper(line, start + 1, curr, counter + 1);

        return max(result, counter + 1);
    }

    else if (curr == DQUOTE && prev != DQUOTE) {
        unsigned int result = task5_helper(line, start + 1, curr, 1);

        return max(result, 1);
    }

    // curr != DQUOTE
    else {
        unsigned int result = task5_helper(line, start + 1, curr, counter);

        return max(result, counter);
    }
}

unsigned int max(unsigned int a, unsigned int b)
{
    return (a > b) ? a : b;
}


// DO NOT WRITE ANYTHING AFTER THIS LINE. ANYTHING AFTER THIS LINE WILL BE REPLACED

const int MAX_LENGTH = 1000;

int main()
{
    int option = 0;
    char line[MAX_LENGTH];

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
