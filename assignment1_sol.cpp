/*
 * COMP2011 (Spring 2020) Assignment 1: Kaprekar's Constant
 *
 * Student name: FILL YOUR NAME HERE
 * Student ID: FILL YOUR STUDENT ID NUMBER HERE
 * Student email: FILL YOUR EMAIL HERE
 *
 * Be reminded that you are only allowed to modify a certain part of this file. Read the assignment description and requirement carefully.
 */
#include <iostream>
#include <math.h>

using namespace std;

const bool ASCENDING = 0;
const bool DESCENDING = 1;
const bool DEBUG = true;

/**
 *  helper function to count #digits in non-negative integer
 */
int countDigit(int number)
{
    int count = 0;
    while (number != 0) {
        number = number / 10;
        ++count;
    }
    return count;
}

/**
 *  Task 1: selectDigit() returns one digit from a given index location of an arbitrary positive integer
 *  The index starts from 0 at the rightmost digit
 *  The function return -1 if index is invalid
 *  Examples: selectDigit(123, 0) = 3, selectDigit(1234, 3) = 1, selectDigit(123, 3) = -1, selectDigit(123, -1) = -1
 */
int selectDigit(int number, int index){

    if(countDigit(number) <= index || index < 0){
      return -1;
    }

    while(index>0){
        number = number/10;
        index--;
    }
    return number%10;
}

/**
 *  Task 2: isRepdigit() checks whether all digits in a positive integer are the same
 *  It returns true if Yes, false otherwise
 *  Examples: isRepdigit(222) = true, isRepdigit(212) = false, isRepdigit(2) = true
 */
bool isRepdigit(int number){

    int count = countDigit(number);
    int pos = count -1;

    if(count==1) return 1;

    for(int i=0;i<pos;i++){
        if(selectDigit(number,i)!=selectDigit(number,i+1)) return 0;
    }
    return 1;
}

/**
 *  Task 3: sortDigit() takes an arbitrary positive integer number,
 *  and returns a new integer with all digits in number arranged in ascending or descending order
 *  Examples: sortDigit(16845, ASCENDING) = 14568, sortDigit(16845, DESCENDING) = 86541
 *  The function should be able to handle number consists 0s
 *  Examples: sortDigit(500, ASCENDING) = 5, sortDigit(100250, DESCENDING) = 521000, sortDigit(100250, ASCENDING) = 125
 */
int sortDigit(int number, bool order){

    int count = countDigit(number);
    int pos = count -1;
    int temp;

    temp = number;

    for (int h=0;h<pos;h++){

        for(int i=1;i<=pos-h;i++){

            int n = temp;
            int temp1 = round(selectDigit(n,i)*pow(10,i)/10);
            int temp2 = round(selectDigit(n,i-1)*pow(10,i-1)*10);

            if(order==ASCENDING){
                if(selectDigit(n,i)>selectDigit(n,i-1)){
                    n = n - (round(selectDigit(n,i)*pow(10,i)) + round(selectDigit(n,i-1)*pow(10,i-1))) ;
                    n = n + temp1 + temp2;
                    temp = n;
                }
            }

            else{
                if(selectDigit(n,i)<selectDigit(n,i-1)){
                    n = n - (round(selectDigit(n,i)*pow(10,i)) + round(selectDigit(n,i-1)*pow(10,i-1))) ;
                    n = n + temp1 + temp2;
                    temp = n;
                }
            }
        }
    }
    number = temp;
    return number;
}

/**
 * Task 4: isKaprekar6174() takes an arbitrary positive integer,
 * and returns the number of steps needed to reach Kaprekar's constant 6174
 * The function returns -1 if the number can't reach 6174
 * when DEBUG is true, calculation details needs to be printed out
 */
int isKaprekar6174(int number, bool debug){
    int n;
    int large, small, diff;
    int step = 0;
    int flag;

    n = number;

    if(countDigit(n)!=4 || isRepdigit(n)==1){

        if(debug)
            cout << "can't reach Kaprekar's constant 6174" << endl;
        return -1;
    }

        while(step < 8){
        step++;
        large = sortDigit(n,DESCENDING);
        small = sortDigit(n,ASCENDING);
        diff = large - small;
        if(debug)
            cout << large << "-" << small << "=" << diff << endl;
        if (countDigit(diff) == (countDigit(n)-1))
            diff *=10;

        if (diff==6174){
                flag = step;
                break;
        }
        n = diff;
        }

    return flag;
}

/**
 *  the helper function to convert number to star
 */

void printStars(int number)
{
    int n=0;

    if(number%50!=0)
        n++;

    number = number/50;
    n = n + number;

    for(int i=0; i<n; i++)
        cout << "*";
    cout << endl;
}

/**
 * Task 5: printStat() bincounts #steps to reach Kaprekar's constant 6174 for numbers in a given range from m to n
 * Then print the bar chart (*)
 * 8 bins are used, which count the numbers with 1 to 7 steps to reach Kaprekar's constant 6174, or fail to do so
 * A * is printed for every 50 counted
 * Print a * if there is remainder
 * Follow strictly the output format described on the website, since your output will be graded by an auto-grader
 * For example if bincount = 350, print 7 stars; bincount = 351, print 8 stars
 * For simplicity, you can assume there is always a valid range
 */
void printStat(int m, int n){

    int bin1 = 0;
    int bin2 = 0;
    int bin3 = 0;
    int bin4 = 0;
    int bin5 = 0;
    int bin6 = 0;
    int bin7 = 0;
    int binFail = 0;

    for(int i=m;i<=n;i++){

        switch(isKaprekar6174(i, false)){
        case(1): bin1++;break;
        case(2): bin2++;break;
        case(3): bin3++;break;
        case(4): bin4++;break;
        case(5): bin5++;break;
        case(6): bin6++;break;
        case(7): bin7++;break;
        case(-1): binFail++;break;
        }
    }

    cout << "1";
    printStars(bin1);
    cout << "2";
    printStars(bin2);
    cout << "3";
    printStars(bin3);
    cout << "4";
    printStars(bin4);
    cout << "5";
    printStars(bin5);
    cout << "6";
    printStars(bin6);
    cout << "7";
    printStars(bin7);
    cout << "-1";
    printStars(binFail);


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
