/*
 * PointSet.cpp
 * 
 */

#include "PointSet.h"
#include <iostream>
using namespace std;


PointSet::PointSet()
   : numPoints{0}
   , points{nullptr}
{
   cout << "Initialized by PointSet's default constructor" << endl;
}

PointSet::PointSet(const Point3D points[], int numPoints)
   : numPoints{numPoints}
   , points{new Point3D[numPoints]}
{
   cout << "Initialized by PointSet's other constructor" << endl;

   for (int i = 0; i < numPoints; ++i)
      this->points[i] = points[i];
}

PointSet::PointSet(const PointSet & s)
   : numPoints{s.numPoints}
   , points{new Point3D[numPoints]}
{
   cout << "Initialized by PointSet's copy constructor" << endl;
   
   for (int i = 0; i < numPoints; ++i)
      points[i] = s.points[i];
}

PointSet::~PointSet()
{
   cout<<"PointSet's destructor is called!" <<endl;

   if (points)
      delete[] points;
}


void PointSet::addPoint(const Point3D& p)
{
   Point3D* newPoints = new Point3D[numPoints + 1];
   for (int i = 0; i < numPoints; ++i)
      newPoints[i] = points[i];
   newPoints[numPoints++] = p;
   std::swap(newPoints, points);
   delete[] newPoints;  // Delete old container (which now resides in newPoints)
}

bool PointSet::contains(const Point3D& p) const
{
   for (int i = 0; i < numPoints; ++i)
      if (points[i].equal(p)) 
         return true;
   return false;
}

void PointSet::removeFirstPoint()
{
   if (numPoints == 0)
   {
      cout << "No points!" << endl;
      return;
   }

   for (int i = 1; i < numPoints; ++i)
      points[i-1] = points[i];
   numPoints--;
}

void PointSet::removeLastPoint()
{
   if (numPoints == 0)
   {
      cout << "No points!" << endl;
      return;
   }

   numPoints--;
}

void PointSet::print() const
{
   if (numPoints == 0)
   {
      cout << "Empty!" << endl;
      return;
   }

   for (int i = 0; i < numPoints; ++i)
   {
      points[i].print();
      cout << endl;
   }
}

