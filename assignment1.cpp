/*
 * COMP2011 (Spring 2020) Assignment 1: Kaprekar's Constant
 *
 * Student name: [redacted]
 * Student ID: [redacted]
 * Student email: [redacted]
 *
 * Note:
 * 1. Add your code ONLY in the corresponding "TODO" area.
 * 2. DO NOT change the function header of the given functions.
 * 3. DO NOT add additional libraries.
 * 4. DO NOT use array, otherwise ZERO marks for the assignment.
 * 5. You can add helper function(s) if needed.
 */
#include <iostream>
#include <math.h>
using namespace std;

#undef DEBUG
const bool ASCENDING = 0;
const bool DESCENDING = 1;
const bool DEBUG = true;

namespace Helper {
    int length(int n) {
        return log10(n) + 1;
    }

    int drop(int n, uint8_t k) {
        return n / int(round(pow(10, k)));
    }

    int take(int n, uint8_t k) {
        return n % int(round(pow(10, k)));
    }
}

/**
 *  Task 1: selectDigit() returns one digit from a given index location of an arbitrary positive integer
 *  The index starts from 0 at the rightmost digit
 *  The function return -1 if index is invalid
 *  Examples: selectDigit(123, 0) = 3, selectDigit(1234, 3) = 1, selectDigit(123, 3) = -1, selectDigit(123, -1) = -1
 */
int selectDigit(int number, int index){
    for (int idx = 0; number != 0; ++idx, number /= 10) {
        if (index == idx)
            return number % 10;
    }
    return -1;
}

/**
 *  Task 2: isRepdigit() checks whether all digits in a positive integer are the same
 *  It returns true if Yes, false otherwise
 *  Examples: isRepdigit(222) = true, isRepdigit(212) = false, isRepdigit(2) = true
 */
bool isRepdigit(int number){
    for (int prev = number % 10; number != 0;
         prev = number % 10, number /= 10) {
        if (prev != number % 10)
            return false;
    }
    return true;
}

/**
 *  Task 3: sortDigit() takes an arbitrary positive integer number,
 *  and returns a new integer with all digits in number arranged in ascending or descending order
 *  Examples: sortDigit(16845, ASCENDING) = 14568, sortDigit(16845, DESCENDING) = 86541
 *  The function should be able to handle number consists 0s
 *  Examples: sortDigit(500, ASCENDING) = 5, sortDigit(100250, DESCENDING) = 521000, sortDigit(100250, ASCENDING) = 125
 */

namespace SortHelper {
    int mergeSort(int number, bool order);
    int insertionSort(int number, bool order);
}

int sortDigit(int number, bool order){
//    return SortHelper::mergeSort(number, order);
    return SortHelper::insertionSort(number, order);
}

/**
 Contains helper class and functions for sorting.
 */
namespace SortHelper
{
    struct DigitStack {
        int n;
        int len;
        
        DigitStack(int n = 0, int len = 0) : n{n}, len{len} {}
        void push(int digit) { n = 10*n + digit; len++; }
        int pop() { const int t = top(); n /= 10; len--; return t; }
        int top() const { return n % 10; }
        bool isEmpty() const { return len == 0; }
    };

    int mergeDigits(DigitStack upper, DigitStack lower, bool order){
        DigitStack number;
        while (!upper.isEmpty() || !lower.isEmpty()) {
            if (lower.isEmpty()) {
                number.push(upper.pop());
            } else if (upper.isEmpty()) {
                number.push(lower.pop());
            } else /*!upper.isEmpty() && !lower.isEmpty()*/ {
                if ((order == ASCENDING && upper.top() < lower.top())
                    || (order == DESCENDING && upper.top() > lower.top())) {
                    number.push(lower.pop());
                } else {
                    number.push(upper.pop());
                }
            }
        }
        return number.n;
    }

    /**
     Merge sort
     */
    int mergeSort_impl(int number, bool order, int length) {
        if (length == 1)
            return number;
            
        const int half = length / 2;
        const int upper_len = length - half;
        const int lower_len = half;
        
        int upper = Helper::drop(number, half);
        int lower = Helper::take(number, half);
        
        //  order should be flipped because merge does a reverse pop/push
        order = !order;
        upper = mergeSort_impl(upper, order, upper_len);
        lower = mergeSort_impl(lower, order, lower_len);
        return mergeDigits(DigitStack{upper, upper_len}, DigitStack{lower, lower_len}, order);
    }

    int mergeSort(int number, bool order) {
        if (number == 0)
            return 0;
        
        int length = Helper::length(number);
        if (length == 1)
            return number;
        
        //  length needed to handle leading 0s
        return mergeSort_impl(number, order, length);
    }


    int insertDigitAt(int number, int digit, int index) {
        int lower = Helper::take(number, index);
        int upper = Helper::drop(number, index);
        int plcval = int(round(pow(10, index)));
        return upper*plcval*10 + digit*plcval + lower;
    }

    int insertionSort(int number, bool order) {
        int sorted = 0;
        int length = 0;
        for (; number; number /= 10) {
            int digit = number % 10;
            for (int i = 0, x = 0; i < length+1; ++i) {
                x = selectDigit(sorted, i);
                if (x == -1) x = 0;
                
                //  found point to insert (maintain sorted order)
                if (i == length
                    || (order == ASCENDING && x <= digit)
                    || (order == DESCENDING && x >= digit)) {
                    
                    sorted = insertDigitAt(sorted, digit, i);
                    length++;
                    break;
                }
            }
        }
        return sorted;
    }
}

/**
 * Task 4: isKaprekar6174() takes an arbitrary positive integer,
 * and returns the number of steps needed to reach Kaprekar's constant 6174
 * The function returns -1 if the number can't reach 6174
 * when DEBUG is true, calculation details needs to be printed out
 * Output format for the calculation details: Each row represents a subtraction calculation
 * There is no space between the number and minus sign. There is no space between the number and equal sign
 */
int isKaprekar6174(int number, bool debug){
    if (isRepdigit(number) || Helper::length(number) != 4) {
        if (debug)
            cout << "can't reach Kaprekar's constant 6174" << endl;
        return -1;
    }
    
    int iter = 0;
    do {
        //  edge case (for 4-digit numbers that arrives at 999, 998, ...; e.g. 1000, 9998)
        if (Helper::length(number) == 3) {
            number *= 10;
        }
        
        int desc = sortDigit(number, DESCENDING);
        int asc = sortDigit(number, ASCENDING);
        number = desc - asc;
        
        if (debug)
            cout << desc << "-" << asc << "=" << number << endl;
        
        iter++;
    } while (number != 6174);
    
    return iter;
}

/**
 * Task 5: printStat() bincounts #steps to reach Kaprekar's constant 6174 for numbers in a given range from m to n (both inclusive)
 * Then print the bar chart (*)
 * For simplicity, you can assume that m and n are positive integers and there is always a valid range
 * 8 bins are used, which count the numbers with 1 to 7 steps to reach Kaprekar's constant 6174, or fail to do so
 * A * is printed for every 50 counted
 * Print a * if there is remainder
 * For example if bincount = 350, print 7 stars; bincount = 351, print 8 stars
 * Output format: For each row, starts with the bin number (i.e. 1 to 7) and immediately followed by the star. Do not leave any spaces between them
 * Output format: The last row (eighth row) of the output represents the failure case (started with -1 and immediately followed by the star, do not leave any spaces between them)
 */

namespace StatsHelper {
    struct BinCounter {
        int count = 0;
        
        void print(int heading) const { cout << heading << string(ceil(count/50.0), '*') << endl; }
        BinCounter operator++(int) { count++; return *this; }
    };
}

void printStat(int m, int n){
    StatsHelper::BinCounter count1, count2, count3, count4, count5, count6, count7, countFail;
    
    for (int i = m; i <= n; ++i) {
        int result = isKaprekar6174(i, !DEBUG);
        switch (result) {
            case 1: count1++; break;
            case 2: count2++; break;
            case 3: count3++; break;
            case 4: count4++; break;
            case 5: count5++; break;
            case 6: count6++; break;
            case 7: count7++; break;
            case -1: countFail++; break;
            default: break;
        }
    }
    
    count1.print(1); count2.print(2); count3.print(3);
    count4.print(4); count5.print(5); count6.print(6);
    count7.print(7); countFail.print(-1);
}

// This is the main function. It is already done. Please DO NOT make any modification.
int main()
{
cout << "Task 1:" << endl;
cout << "selectDigit(896543,0) = " << selectDigit(896543,0) << endl;
cout << "selectDigit(896543,5) = " << selectDigit(896543,5) << endl;
cout << "selectDigit(896543,-1) = " << selectDigit(896543,-1) << endl;
cout << "selectDigit(896543,6) = " << selectDigit(896543,6) << endl;

cout << endl << "Task 2:" << endl;
cout << "isRepdigit(2999) " << boolalpha << isRepdigit(2999) << endl;
cout << "isRepdigit(888888) " << boolalpha << isRepdigit(888888) << endl;
cout << "isRepdigit(1) " << boolalpha << isRepdigit(1) << endl;

cout << endl << "Task 3:" << endl;
cout << "sortDigit(54321, ASCENDING) = " << sortDigit(54321, ASCENDING) << endl;
cout << "sortDigit(794621, ASCENDING) = " << sortDigit(794621, ASCENDING) << endl;
cout << "sortDigit(794621, DESCENDING) = " << sortDigit(794621, DESCENDING) << endl;
cout << "sortDigit(100250, ASCENDING) = " << sortDigit(100250, ASCENDING) << endl;
cout << "sortDigit(100250, DESCENDING) = " << sortDigit(100250, DESCENDING) << endl;
cout << "sortDigit(500, ASCENDING) = " << sortDigit(500, ASCENDING) << endl;

cout << endl << "Task 4:" << endl;
cout << "isKaprekar6174(546, DEBUG) = " << isKaprekar6174(546, !DEBUG) << endl;
isKaprekar6174(546, DEBUG);
cout << "isKaprekar6174(18604, DEBUG) = " << isKaprekar6174(18604, !DEBUG)<< endl;
isKaprekar6174(18604, DEBUG);
cout << "isKaprekar6174(8888, DEBUG) = " << isKaprekar6174(8888, !DEBUG) << endl;
isKaprekar6174(8888, DEBUG);
cout << "isKaprekar6174(2894, DEBUG) = " << isKaprekar6174(2894, !DEBUG)<< endl;
isKaprekar6174(2894, DEBUG);
cout << "isKaprekar6174(6174, DEBUG) = " << isKaprekar6174(6174, !DEBUG)<< endl;
isKaprekar6174(6174, DEBUG);

cout << endl << "Task 5: " << endl;
cout << "Statistic for range from 1000 to 9999" << endl;
printStat(1000,9999);
cout << "Statistic for range from 500 to 10500" << endl;
printStat(500,10500);

return 0;
}
