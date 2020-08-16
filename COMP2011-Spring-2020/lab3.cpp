//
//  lab3.cpp
//  Copyright © 2020 TrebledJ. All rights reserved.
//

#include <iostream>
#include <iomanip>      //  std::setprecision
#include <random>       //  std::mt19937, std::uniform_real_distribution<>, std::random_device
#include <tuple>
#include <thread>       //  std::this_thread::sleep_for
#include <limits>       //  std::numeric_limits


using namespace std;

using Point = tuple<double, double>;
using ld = long double;

class RandomPointGenerator {
public:
    //  x/y are coordinates on the first quadrant
    RandomPointGenerator(double x, double y)
        : x_dis{0.0, x}, y_dis{0.0, y}
    {}
    
    Point get() {
        return make_tuple(x_dis(gen), y_dis(gen));
    }
    
private:
    static mt19937 gen;
    uniform_real_distribution<> x_dis;
    uniform_real_distribution<> y_dis;
};

mt19937 RandomPointGenerator::gen{random_device()()};


/**
 Finds the area of an ellipse using the Monte Carlo integration method.
 Accuracy can reach at least 3 s.f. for 1,000,000 iterations.
 
 Advantages:
 * O(n) wrt iterations    (∩ ﾟᎲ ﾟ)⊃━☆ﾟ.*
 * Takes the same amount of time for different a, b.
 * Can be generalised for other functions
 * For high iterations, accurate to 3+ s.f.
 
 Disadvantages:
 * Inaccurate, relatively large percentage error
 */
long double monte_carlo_ellipse_integration(double a, double b, uint32_t iterations) {
    RandomPointGenerator rpg{a, b};
    const double a2 = a*a;
    const double b2 = b*b;
    const double a2b2 = a2 * b2;
    
    uint32_t hit = 0;
    for (uint32_t i = 0; i < iterations; ++i) {
        const auto [x, y] = rpg.get();
        if (x*x*b2 + y*y*a2 <= a2b2)
            hit++;
    }
    
    cout << "> hit/iterations: " << hit << "/" << iterations << endl;
    
    //  multiply area of first quadrant sample space by
    //  experimental probablity of hitting ellipse
    return 4.0 * (a*b * (long double)hit/iterations);
}

inline bool is_inside_ellipse(ld a, ld b, ld x, ld y) {
    return x*x*b*b + y*y*a*a <= a*a*b*b;
}

/**
 Calculates the area of an ellipse by using middle reimann sums in two dimensions.
 
 Advantages:
 * Very accurate
 
 Disadvantages:
 * O(mn) wrt a/precision, b/precision     ᗒ ͟ʖᗕ
 * Takes longer for smaller precision or larger a, b
 */
ld ellipse_double_summation(ld a, ld b, ld precision) {
    //  calculate area outside ellipse and subtract from total
    
    const ld precision_area = precision * precision;
    const ld a2 = a*a;
    const ld b2 = b*b;
    const ld a2b2 = a2 * b2;
    ld total = a*b;
    
    for (ld x = a - precision/2; x >= 0.0; x -= precision) {
        for (ld y = b - precision/2; y >= 0.0; y -= precision) {
            if (x*x*b2 + y*y*a2 > a2b2) {
                total -= precision_area;
            } else {
                break;
            }
        }
    }
    
    return 4.0 * total;
}

/**
 Calculates the area of an ellipse using a middle reimann sum.
 Uses binary search to determine appropriate rectangles for the sum.
 
 Advantages:
 * O(m log n) wrt a/precision, b/precision
 * Accurate
 
 Disadvantages:
 * Takes longer for smaller precision or larger a, b
 */
ld ellipse_reimann_sum_bs(ld a, ld b, ld precision) {
    const ld a2 = a*a;
    const ld b2 = b*b;
    const ld a2b2 = a2 * b2;
    ld total = 0.0;
    
    for (ld x = precision/2; x < a; x += precision) {
        ld y_low = 0.0;
        ld y_high = b;
        
        while (y_low < y_high && y_high - y_low > precision * 1e-6) {
            ld y_mid = (y_high + y_low)/2;
            
            bool is_lower_inside = x*x*b2 + y_mid*y_mid*a2 <= a2b2;
            bool is_upper_inside = x*x*b2 + (y_mid+precision)*(y_mid+precision)*a2 <= a2b2;
            
            if (is_lower_inside && !is_upper_inside) {
                //  (x, y_mi) is inside ellipse
                total += y_mid * precision; //  add the rectangle
                break;
            } else if (is_lower_inside && is_upper_inside) {
                //  guessed too low
                y_low = y_mid;
            } else if (!is_lower_inside && !is_upper_inside) {
                //  guessed too high
                y_high = y_mid;
            }
        }
    }
    
    return 4.0 * total;
}

/**
 Advantages:
 * O(1) wrt a, b       (づ◔ ͜ʖ◔)づ
 */
ld calculate_ellipse_area(ld a, ld b) {
    static constexpr ld PI = 3.141592653589793238462643;
    return PI * a * b;
}


void delay_short() { this_thread::sleep_for(200ms); }
void delay_long() { this_thread::sleep_for(1000ms); }
void ignore_all() { cin.ignore(numeric_limits<streamsize>::max(), '\n'); }

void wait_for_enter() {
    cout << endl;
    delay_long();
    cout << "(enter to continue) ";
    cin.get();
    delay_short();
    cout << endl;
    delay_short();
    cout << endl;
    delay_short();
}

void show_actual_value(ld a, ld b, ld val) {
    ld actual = calculate_ellipse_area(a, b);
    cout << endl;
    cout << "> actual value: " << actual << endl;
    
    ld error = fabs(val - actual);
    ld percentage_error = round(error / actual * 10'000) / 100;
    
    cout << "> error: " << error;
    if (percentage_error < 0.005) {
        cout << " (< 0.005%)" << endl;
    } else {
        cout << " (" << percentage_error << "%)" << endl;
    }
    cout << endl;
    
    wait_for_enter();
}

bool assert_positive(ld value) {
    if (value <= 0) {
        cout << endl << "!! value should be positive (x > 0) !!" << endl << endl;
        wait_for_enter();
        return false;
    }
    return true;
}


int main() {
    cout << setprecision(12);
    while (1) {
        char opt;
        
        cout << "Select a calculation method:" << endl;
        cout << "  1/A - Double Summation (Brute Force)" << endl;
        cout << "  2/B - Reimann Sum (with Binary Search)" << endl;
        cout << "  3/M - Monte Carlo Integration" << endl;
        cout << "  0/Q - Quit" << endl;
        
        cout << ">>> ";
        cin >> opt; ignore_all();
        
        switch (opt) {
            case '1':
            case 'a':
            case 'A': {
                ld precision;
                ld a, b;
                
                cout << endl << "Double Summation:";
                cout << endl << "Precision? (p > 0, real)" << endl << ">>> ";
                cin >> precision; ignore_all();
                if (!assert_positive(precision)) continue;
                
                cout << "a? b?" << endl << ">>> ";
                cin >> a >> b; ignore_all();
                a = fabs(a);
                b = fabs(b);
                
                ld val = ellipse_double_summation(a, b, precision);
                cout << endl << "> double summation: " << val << endl;
                show_actual_value(a, b, val);
            } break;
            case '2':
            case 'b':
            case 'B': {
                ld precision;
                ld a, b;
                
                cout << endl << "Reimann Sum:";
                cout << endl << "Precision? (p > 0, real)" << endl << ">>> ";
                cin >> precision; ignore_all();
                if (!assert_positive(precision)) continue;
                
                cout << "a? b?" << endl << ">>> ";
                cin >> a >> b; ignore_all();
                a = fabs(a);
                b = fabs(b);
                
                ld val = ellipse_reimann_sum_bs(a, b, precision);
                cout << endl << "> reimann: " << val << endl;
                show_actual_value(a, b, val);
            } break;
            case '3':
            case 'm':
            case 'M': {
                ld iterations;
                ld a, b;
                
                cout << endl << "Monte Carlo Integration:";
                cout << endl << "Iterations? (n > 0, integer)" << endl << ">>> ";
                cin >> iterations; ignore_all();
                if (!assert_positive(iterations)) continue;

                cout << "a? b?" << endl << ">>> ";
                cin >> a >> b; ignore_all();
                a = fabs(a);
                b = fabs(b);
                
                ld mc = monte_carlo_ellipse_integration(a, b, (uint32_t)iterations);
                cout << endl << "> monte carlo: " << mc << endl;
                show_actual_value(a, b, mc);
            } break;
            case '0':
            case 'q':
            case 'Q': goto out;
            default:
                cout << "'" << opt << "' is not a recognised option." << endl << endl;
                break;
        }
    }
    
    out:
    cout << "Goodbye o/" << endl;
}
