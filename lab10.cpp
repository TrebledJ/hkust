//
//  COMP2011
//  lab10.cpp
//  Copyright © 2020 TrebledJ. All rights reserved.
//

#include <iostream>
#include <cstdlib>
using namespace std;

//  Provided print array function
void print_array(const int* array, int size)
{
    for (int i = 0; i < size; i++)
        cout << array[i] << ' ';
    cout << endl;
}


//  Find missing number function
//  You should return the missing number
int missing_number(const int* nums, int size)
{
    unsigned long long total = size*(size+1)/2;  //  sum of numbers in [1, size+1]
    for (int i = 0; i < size; ++i)
        total -= nums[i];
    return int(total);
}


using precise = long double;

//  returns an int type in the range [0, n)
unsigned randint(unsigned n) { return n ? (rand() % n) : 0; }

void shuffle_array(int* arr, int size)
{
    //  https://en.wikipedia.org/wiki/Fisher–Yates_shuffle#The_modern_algorithm
    for (int i = size-1; i >= 0; --i)
        std::swap(arr[randint(i)], arr[i]);
}

//  Generate the shuffled array missing one number
int* generate_array(int size)
{
    //  create array
    int* arr = new int[size];
    int leave_out = randint(size+1);
    for (int i = 0; i < size+1; ++i)
    {
        if (i == leave_out)
            continue;
        
        arr[(i > leave_out ? i-1 : i)] = i;
    }
    
    //  shuffle array
    shuffle_array(arr, size);
    
    return arr;
}

//  You should first generate array and then find the missing number
//  You should free all allocated memory and return the missing number (which is used to determine the length of the array in the next iteration)
int solve(int size, int iter)
{
    int* arr = generate_array(size);
    int x = missing_number(arr, size);
    cout << endl;
    print_array(arr, size);
    cout << "x" << iter << ": " << x << endl;
    delete[] arr;
    return x;
}

int get_int_in_range(int low, int high, std::string const& prompt)
{
    int n;
    while (1)
    {
        cout << prompt << endl;
        if (!(cin >> n))
        {
            cout << "input rejected: parse error" << endl;
            cin.clear();
        }
        else if (!(low <= n && n <= high))
            cout << "input rejected: out of range" << endl;
        else
            break;
        
        cin.ignore(1000, '\n');
    }
    return n;
}


constexpr int MAX_ITERATIONS = 5;
int main()
{
    srand(time(0));
    int size = get_int_in_range(2, 100, "(length: [2, 100]) >>> ");
    int iter = get_int_in_range(1, MAX_ITERATIONS, "(iterations: [1, 5]) >>> ");

    int x[MAX_ITERATIONS] = {0};
    for (int i = 0; i < iter; ++i)
    {
        size += (x[i] = solve(size, i));
    }
    cout << endl;
    for (int i = 0; i < iter; ++i)
        cout << "x" << i << ": " << x[i] << endl;
    
    return 0;
}
