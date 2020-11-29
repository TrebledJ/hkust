#ifndef SET_H
#define SET_H

#include <iostream>
#include <sstream>

using namespace std;
ostream& operator<<(ostream& os, pair<string, int> value);


template<typename T>
class set {
private:
	T* elements { nullptr };
	int size { 0 };
	string str() const {
		ostringstream os;
		os << "{";
		if (size != 0) {
			os << elements[0];
			for (int i = 1; i < size; i++) {
				os << "," << elements[i];
			}
		}
		os << "}";
		return os.str();
	}

public:
	set() = default;
	set(const T * arr, int size) :
			size(size) {
		elements = new T[size];
		for (int i = 0; i < size; i++)
			elements[i] = arr[i];
	}
	~set() {
		delete[] elements;
	}

	// Return the cardinality of the set, i.e., the size
	int cardinality() const {
		return size;
	}

	// Return the element's index if it exists. Otherwise return -1
	int findElement(const T& element) const {
		int pos = -1;
		for (int i = 0; i < size; i++) {
			if (elements[i] == element) {
				pos = i;
				break;
			}
		}
		return pos;
	}

	// Return true if the element exists, otherwise return false
	bool isExist(const T& element) const {
		int pos = findElement(element);
		bool existence = (pos == -1) ? false : true;
		return existence;
	}

	// Allocate a new array of size "size+1", copy all the existing elements over and
	// add the new element to the last position of the new array
	void addElement(const T& element) {

		if (!isExist(element)) {
			T* newElements = new T[size + 1];
			for (int i = 0; i < size; i++)
				newElements[i] = elements[i];
			newElements[size] = element;
			delete[] elements;
			elements = newElements;
			size++;
		}
	}

	// Check if the element is in the set.
	// If it is in the set, allocate a new array and copy all the existing elements over except
	// the element to be deleted.
	// Return true if the element is found and deleted. Otherwise return false
	bool removeElement(const T& element) {
		if (isExist(element)) {
			int pos = findElement(element);
			T* newElements = new T[size - 1];
			int count = 0;
			for (int i = 0; i < size; i++) {
				if (i != pos)
					newElements[count++] = elements[i];
			}
			delete[] elements;
			elements = newElements;
			size--;
			return true;
		} else
			return false;
	}

	/////////////////////////////////////////////////////////////////////////
	//                                                                     //
	//      	You should only modify the code after this line            //
	//                                                                     //
	/////////////////////////////////////////////////////////////////////////

	// Copy constructor: Perform deep copying
	// Hint: Make use of assignment operator function `operator=`
	set(const set& rhs)
		: elements{new T[rhs.size]}
		, size{rhs.size}
	{
		for (int i = 0; i < rhs.size; i++)
			elements[i] = rhs.elements[i];
	}

	// Overload `operator+` to support union operation of two set objects
	set operator+ (const set& rhs) const
	{
		set newset = *this;
		for (int i = 0; i < rhs.size; i++)
			newset.addElement(rhs.elements[i]);
		return newset;
	}

	// Overload `operator*` to support intersect operation of two set objects
	set operator* (const set& rhs) const
	{
		set newset = rhs;
		for (int i = 0; i < newset.size;)
		{
			if (!isExist(newset.elements[i]))
				newset.removeElement(newset.elements[i]);
			else
				i++;
		}
		return newset;
	}

	// Overload assignment operator function `operator=`
	// Note: Deep copying is required
	set& operator= (const set& rhs)
	{
		delete[] elements;
		elements = new T[rhs.size];
		for (int i = 0; i < rhs.size; i++)
			elements[i] = rhs.elements[i];
		return *this;
	}

	// Overload equality operator function `operator==`
	bool operator== (const set& rhs) const
	{
		if (size != rhs.size) return false;
		for (int i = 0; i < size; i++)
			if (!rhs.isExist(elements[i]))
				return false;
		return true;
	}

	// Using the "friend" keyword, declare stream insertion operator function `operator<<`
	friend ostream& operator<< (ostream& os, const set& s)
	{
		return os << s.str();
	}

};

#endif
