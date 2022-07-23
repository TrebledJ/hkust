#include <algorithm>
#include <cstring>
#include <iostream>
#include <stack>
#include <vector>


using namespace std;


// Checks whether a given string contains only math operators.
inline bool is_operator(const string &str)
{
    return str.find_first_not_of("+-*/") == string::npos;
}
 
// Checks whether a given string contains only digits.
inline bool is_digits(const string &str)
{
    return str.find_first_not_of("0123456789") == string::npos;
}

// Returns a vector storing the numbers and operators of the input formula.
vector<string> store_the_formula(const string &formula)
{
    vector<string> v;
    std::size_t prev = 0, i = -1;
    while ((i = formula.find(" ", i+1)) != string::npos)
    {
        std::string sub = formula.substr(prev, i - prev);
        if (!sub.empty())
            v.push_back(sub);
        prev = i+1;
    }
    std::string sub = formula.substr(prev);
    if (!sub.empty())
        v.push_back(sub);
    return v;
}


using Operation = int (*)(int, int);
inline int add(int a, int b) { return a + b; }
inline int sub(int a, int b) { return a - b; }
inline int mul(int a, int b) { return a * b; }
inline int divv(int a, int b) { return a / b; }
inline Operation operation(char op)
{
    switch (op)
    {
    case '+': return add;
    case '-': return sub;
    case '*': return mul;
    case '/': return divv;
    default: return nullptr;
    }
}

// Prints the calulation steps of the given formula. You must use iterators to traverse the formula 
// vector and a stack and to evaluate the formula, as explained in the lab description. First you need to determine 
// whether a formula is in Polish or Reverse Polish notation.
// Hint: You may use is_digits and is_operators to tell apart Polish Notation and Reverse Polish Notation.
void calculation_steps(vector<string> seq)
{
    if (seq.empty())
    {
        cout << "Sequence is empty!" << endl;
        return;
    }

    stack<int> stk;

    // Regular: Scan from right to left. Reversed: Scan from left to right.
    const bool reversed = is_digits(seq[0]);
    if (!reversed)
        reverse(seq.begin(), seq.end());
    
    // for (const string& str : seq)
    using It = vector<string>::const_iterator;
    for (It it = seq.begin(); it != seq.end(); ++it)
    {
        const string& str = *it;

        if (is_digits(str))
        {
            int i = stoi(str);
            stk.push(i);
            cout << "push " << i << " to the stack." << endl;
        }
        else
        {
            if (stk.size() < 2)
            {
                cout << "Not enough elements in stack!" << endl;
                return;
            }

            // Pop last two from stack and perform op.
            int t2 = stk.top(); stk.pop();
            cout << "pop " << t2 << " from the stack." << endl;
            int t1 = stk.top(); stk.pop();
            cout << "pop " << t1 << " from the stack." << endl;

            if (!reversed)
                swap(t1, t2);
            
            const char op = str[0];
            const int res = operation(op)(t1, t2);
            cout << t1 << op << t2 << "=" << res << endl;

            stk.push(res);
            cout << "push " << res << " to the stack." << endl;
        }
    }

    if (stk.empty())
    {
        cout << "No elements in stack after calculation!" << endl;
        return;
    }

    cout << "Result: " << stk.top() << endl;
}

// Calculates and prints the result of evaluating a formula in Polish or Inverse Polish format. The formula may contain +-*/ operators.
void calculation_result(const string &formula)
{
    // Transform the input string into the one with format of the 
    // corresponding notation and store it in a vector.
    vector<string> sequence = store_the_formula(formula);

    // Print the notation stored in the vector.
    cout << "Formula:" << " ";
    for(vector<string>::const_iterator it = sequence.begin(); it != sequence.end(); ++it) 
    {
        cout << *it << " ";
    }

    // Calculate the result and print it out.
    cout << endl << "Calculation steps: " << endl;
    calculation_steps(sequence);
}

void test(string formula)
{
    static int i = 1;
    if (i > 1)
        cout << endl;
    cout << "Test " << (i++) << ":" << endl;
    calculation_result(formula);
}

int main()
{
    test("5 1 2 + 4 * + 3 -");
    test("9 3 1 - 3 * 10 2 / + +");
    test("+ + 2 * 3 - 10 4 / 8 4");
    test("- * 2 + 1 5 4");
    return 0;
}
