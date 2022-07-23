#include <iostream>
using namespace std;

struct Point {
    double x;
    double y;
    double z;
};

/*******************************************************************************
 *  * The functions we provided
 *  *******************************************************************************/
void generate_point(Point pts[], int size);

double get_distance(Point p1, Point p2);

void output_point(Point p);

void output_points(Point pts[], int size);

/*******************************************************************************
 *  * The functions you need to implement
 *  *******************************************************************************/
Point get_closest_point(Point pts[], Point p, int size);

Point mean_point(const Point &a, const Point &b);

void merge_point(Point pts[], int &size, int pid1, int pid2);

Point mean_all_points(Point pts[], int size);
/*******************************************************************************
 *  * Helper Function
 *   * Remark: You might need to define the helper function for convenience
 *   *******************************************************************************/
void merge_single(Point pts[], int &size);

void merge_multiple(Point pts[], int &size, int n);
/*******************************************************************************
 *  Implementation: These function might help you
 *  *******************************************************************************/

// Point pts[] = {};

void generate_point(Point pts[], int size) {
    double x = 3, y = 8, z = 2;
    for (int i = 0; i < size; i++, x *= 1.9, y *= 0.9, z *= 1.1) {
        pts[i].x = x;
        pts[i].y = y;
        pts[i].z = z;
    }
}

double get_distance(Point p1, Point p2) {
    double distance = 0;
    distance += (p1.x - p2.x) * (p1.x - p2.x);
    distance += (p1.y - p2.y) * (p1.y - p2.y);
    distance += (p1.z - p2.z) * (p1.z - p2.z);
    return distance;
}

void output_point(Point p) {
    cout << p.x << " " << p.y << " " << p.z << endl;
}

void output_points(Point pts[], int size) {
    for (int i = 0; i < size; i++) {
        output_point(pts[i]);
    }
}

/*******************************************************************************
 *  Implementation: Define your function here
 *  *******************************************************************************/
Point get_closest_point(Point pts[], Point p, int size) {
    double min_dis = get_distance(p, pts[0]);
    int index = 0;
    for (int i = 1; i < size; i++) {
        double dis = get_distance(p, pts[i]);
        if (min_dis > dis) {
            min_dis = dis;
            index = i;
        }
    }

    return pts[index];
}

Point mean_point(const Point &a, const Point &b) {
    Point p;
    p.x = (a.x + b.x) / 2;
    p.y = (a.y + b.y) / 2;
    p.z = (a.z + b.z) / 2;

    return p;
}

Point mean_all_points(Point pts[], int size) {
    double sx = 0.0, sy = 0.0, sz = 0.0;
    for (int i = 0; i < size; i++) {
        sx += pts[i].x;
        sy += pts[i].y;
        sz += pts[i].z;
    }
    Point p;
    p.x = sx / size;
    p.y = sy / size;
    p.z = sz / size;

    return p;
}

void merge_point(Point pts[], int &size, int pid1, int pid2) {
    Point p = mean_point(pts[pid1], pts[pid2]);
    pts[pid1] = p;
    pts[pid2] = pts[size - 1];
    size--;
}



void merge_single(Point pts[], int &size) {
    int pid1 = 0, pid2 = 1;
    double min_dis = get_distance(pts[0], pts[1]);
    for (int i = 0; i < size; i++) {
        for (int j = i + 1; j < size; j++) {
            double dis = get_distance(pts[i], pts[j]);
            if (min_dis > dis) {
                min_dis = dis;
                pid1 = i;
                pid2 = j;
            }
        }
    }
    merge_point(pts, size, pid1, pid2);
}

void merge_multiple(Point pts[], int &size, int n) {
    for (int i = 0; i < n; i++) {
        merge_single(pts, size);
    }
}

/*******************************************************************************
 *  Implementation: Define the entry of the program
 *  *******************************************************************************/
int main() {
    int size;
    Point p;

    cout << "Please input the number of the points to generate" << endl;
    cin >> size;
    Point* pts = new Point[size];
    generate_point(pts, size);

    
    cout << size << " points have been generated successfully" << endl;
    output_points(pts, size);
    cout << endl;

    /*Task 1: Find the closest point*/
    cout << "Task 1: Find the closest point" << endl;
    cout << "Input your 3D point" << endl;
    cin >> p.x >> p.y >> p.z;
    Point q = get_closest_point(pts, p, size);
    cout << "The closest point is " << endl;
    output_point(q);
    cout << endl;

    /*Task 2: merge closest points*/
    cout << "Task 2: merge closest points" << endl;
    int merge_num = size - 3;
    merge_multiple(pts, size, merge_num);
    cout << "The remaining points are " << endl;
    output_points(pts, size);
    cout << endl;

    Point mean = mean_all_points(pts, 3);
    cout << "The mean point of remaining points is " << endl;
    output_point(mean);
    cout << endl;

    delete [] pts;

    return 0;
}
