/* COMP2011 Spring 2020
 * Lab 3 - Area of Ellipse
 * Suggested Solution
 */

#include <iostream>
#include <iomanip>

using namespace std;

// returns true if the point is inside the ellipse; otherwise, returns false
bool in_ellipse(long double x, long double y, long double square_a, long double square_b)
{
    return (x*x) / square_a + (y*y) / square_b <= 1.0;
}

int main()
{
    // dimension of each square on the grid
    long double precision = 0.0000001;
    cout << "Please enter the precision: ";
    cin >> precision;

    // width & height of ellipse
    long double a = 1, b = 1;
    cout << "Please enter the \'a\' and \'b\' of the ellipse: ";
    cin >> a >> b;

    long double square_a = a * a;
    long double square_b = b * b;

    // adding grid of squares to the coordinate system,
    // # squares on the X-axis
    unsigned long long num_X = a/precision;
    // # squares on the Y-axis
    unsigned long long num_Y = b/precision;

    // # squares covered by ellipse in the 1st quadrant
    unsigned long long quadrant_squares = 0;

    // # squrares in a row inside ellipse
    unsigned long long row_squares = num_X;

    // current position, starting from bottom right of the 1st quadrant
    long double x = a - precision/2, y = precision/2; // (x, y): center of a square

    // counting the squares covered by ellipse in the 1st quadrant row-by-row
    for (unsigned long i = num_Y; i>0; i--)
    {
       // checking whether squares of a row are outside the ellipse
       while (!in_ellipse(x, y, square_a, square_b))
       {
           x -= precision; // move leftwards by one square
           row_squares --; // subtract the outside square
       }
       quadrant_squares += row_squares; // update the total # of squares in a quadrant
                                        // with the # of squares inside ellipse for the row
       y += precision;  // move upwards by one row
    }
    // CLEVER TRICKS are used in the above nested loop:
    // - inner loop: stops as soon as a square is inside ellipse, no need to count further
    // - outer loop: when moving from row i to row i+1, x does not restart from (a - precision/2),
    //               it starts from the final value of x in row i

    cout.precision(9);
    // area of ellipse = 4 x # squares in 1st quadrant x area of each square
    cout << (quadrant_squares * precision * precision * 4) << endl;

    return 0;
}
