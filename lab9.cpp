//
//  lab9.cpp
//  Copyright © 2020 TrebledJ. All rights reserved.
//

/*
 * COMP2011 Spring 2020
 * Lab 9: Clustering
 */

#include "lab_helpers.hpp"

#include <iostream>

#include <algorithm>
#include <utility>
#include <vector>

using namespace std;


//  and I like to make a class out of things >.>
struct Point
{
    double x;
    double y;
    double z;
    
    Point() : x{0}, y{0}, z{0}
    {}
    Point(double x, double y, double z) : x{x}, y{y}, z{z}
    {}
    
    Point operator+ (Point const& rhs) const
    {
        return {x+rhs.x, y+rhs.y, z+rhs.z};
    }
    Point operator/ (double n) const
    {
        return {x/n, y/n, z/n};
    }
    
    Point& operator+= (Point const& rhs)
    {
        x += rhs.x; y += rhs.y; z += rhs.z;
        return *this;
    }
    
    friend istream& operator>> (istream& is, Point& p)
    {
        is >> p.x >> p.y >> p.z;
        return is;
    }
    
    friend ostream& operator<< (ostream& os, Point const& p)
    {
        os << p.x << " " << p.y << " " << p.z;
        return os;
    }
};

//  so I like to overload operator<< so that we can use the type naturally with cout
template<class T>
ostream& operator<< (ostream& os, vector<T> const& v)
{
    for (auto const& e : v)
        os << e << endl;
    return os;
}

//  there's a nickname for this class, it's called thomas
class Engine
{
    Scanner sc;
    Printer print;
public:
    int run()
    {
        auto size = sc.get_a<unsigned>("(#points) >>> ");
        vector<Point> pts = generate_points(size);

        print(size, "points have been generated successfully");
        print(pts);
        print();

        /* Task 1: Find */
        print("Task 1: Find Closest Point");
        
        auto p = sc.get_a<Point>("(<x> <y> <z>) >>> ");
        auto q = get_closest_point(pts, p);
        print("The closest point is", q);
        print();

        /* Task 2: Merge */
        print("Task 2: Merge Closest Points");
        int merge_num = size - 3;
        assert(merge_num > 0);
        merge_multiple(pts, merge_num);
        print("The remaining points are");
        print(pts);
        print();

        Point mean = mean_points(pts);
        print("The mean point of remaining points is", mean);
        print();
        
        return 0;
    }
    
    Point get_closest_point(vector<Point> const& pts, Point const& p)
    {
        assert(!pts.empty());
        return *min_element(pts.begin(), pts.end(),
                            [&](auto const& a, auto const& b) { return get_distance(a, p) < get_distance(b, p); }
                            );
    }
    
    /// returns indices
    pair<int, int> get_nearest_points(vector<Point> const& pts)
    {
        pair<int, int> closest;
        double closest_d = 1e24;
        //  O(n^2), the spitting image of inefficency -_-
        for (auto it = pts.begin(); it != pts.end(); ++it)
            for (auto jt = it+1; jt != pts.end(); ++jt)
                if (auto d = get_distance(*it, *jt); d < closest_d)
                {
                    closest = make_pair(it - pts.begin(), jt - pts.begin());
                    closest_d = d;
                }
        return closest;
    }

    //  Compute mean of point array and return the mean point
    Point mean_points(vector<Point> const& pts)
    {
        Point sum;
        for (auto const& p : pts)
            sum += p;
        return sum / pts.size();
    }

    //  merge closest pair of points, update the pts array and size
    void merge_single(vector<Point>& pts)
    {
        assert(pts.size() >= 2);
        auto [i, j] = get_nearest_points(pts);
        Point mean = mean_points({pts[i], pts[j]});
        pts[j] = mean;
        pts.erase(pts.begin() + i);
    }
    
    void merge_multiple(vector<Point>& pts, int n)
    {
        for (auto i = 0; i < n; ++i)
            merge_single(pts);
    }
    
private:
    //  generate a vector points (seed is hard-coded >.>)
    static vector<Point> generate_points(unsigned size)
    {
        vector<Point> pts(size);
        generate(pts.begin(), pts.end(), [x=3.f, y=8.f, z=2.f]() mutable { Point p{x, y, z}; x *= 1.9; y *= 0.9; z *= 1.1; return p; });
        return pts;
    }

    //  returns the euclidean distance between two points
    static double get_distance(Point const& p1, Point const& p2)
    {
        double distance = 0;
        distance += (p1.x - p2.x) * (p1.x - p2.x);
        distance += (p1.y - p2.y) * (p1.y - p2.y);
        distance += (p1.z - p2.z) * (p1.z - p2.z);
        return distance;
    }

};


int main() {
    Engine engine;
    return engine.run();
}
