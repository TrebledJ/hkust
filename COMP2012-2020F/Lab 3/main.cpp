#include <iostream>
#include "PointSet.h"
using namespace std;

int main()
{
	cout << "PointSet1:" << endl;
	PointSet set1; // Initialized by the default constructor
	set1.print();

	cout << "removeFirstPoint:" << endl;
	set1.removeFirstPoint();

	cout << endl;
	Point3D points[4] = {Point3D(0, 0, 0), Point3D(1.5, 2, 0), Point3D(0.5, 4, 0), Point3D(1.5, 2.5, 3) };
	cout << "PointSet2:" << endl;
	PointSet *set2 = new PointSet(points, 4); // Initialized by the other constructor
	set2->print();

	cout << endl;
	cout << "PointSet3:" << endl;
	PointSet set3(*set2); // Initialized by the copy constructor
	set3.print();
	cout << "addPoint:" << endl;
	set3.addPoint(Point3D(7, 9.4, 6.3));
	set3.print();
	cout << "addPoint:" << endl;
	set3.addPoint(Point3D(3, 1.7, 0));
	set3.print();

	cout << "removeFirstPoint:" << endl;
	set3.removeFirstPoint();
	set3.print();

	cout << "removeLastPoint:" << endl;
	set3.removeLastPoint();
	set3.print();

	/* Setting the boolalpha format flag so that bool values will be
	* displayed in their textual representation: either true or false,
	* instead of integral values.
	*/
	cout << boolalpha;
	cout << "Does PointSet3 have the point (1, 2, 3)? "
	    << set3.contains(Point3D(1, 2, 3)) << endl;
	cout << "Does PointSet3 have the point (1.5, 2.5, 3)? "
	    << set3.contains(Point3D(1.5, 2.5, 3)) << endl;
	cout << endl;

	delete set2;

	return 0;
}
