/*
 * PointSet.h
 * The header file for the class PointSet
 */

#ifndef POINTSET_H
#define POINTSET_H

#include "Point3D.h"
class PointSet
{
	public:
		PointSet();	//Default constructor: set points to nullptr and numPoints to 0
		PointSet(const PointSet& s); //Copy constructor
		PointSet(const Point3D points[], int numPoints); // Other constructor: to construct with the given points; see sample output
		~PointSet(); //Destructor

		void addPoint(const Point3D& p);	//add a 3D point p(x,y,z) to the set; you must resize the array, as the array should be always just big enough to contain all points
		void removeFirstPoint(); //remove the first point (the one with the smallest index) from the set; output (cout) the message "No points!" and do nothing else if the set has no points at all
		void removeLastPoint(); //remove the last point (the one with the largest index) from the set; output (cout) the message "No points!" and do nothing else if the set has no points at all
		bool contains(const Point3D& p) const;	//return true if the given 3D point p(x,y,z) is in the set; return false otherwise; You may want to use the "equal" member function of Point3D
		void print() const;	// print the list of 3D points in the set: simply print the points one by one from the first one to the last, with one point per line, see sample output. If there is no point in the set, print "Empty!"

	private:
		int numPoints; //number of points in this set
		Point3D *points; //this array must be just big enough to contain all points, therefore its size would be simply "numPoints"

};

#endif
