#include <set>
#include <algorithm>
#include <string>
#include <iostream>
#include <list>
#include <map>
using namespace std;


void printSetUsingIterator(const set<string>& s) {
	cout << "{";
	if (!s.empty())
	{
		// cout << *s.begin();
		copy(s.begin(), s.end(), ostream_iterator<string>(cout, ","));
		// for (auto it = std::next(s.begin()); it != s.end(); ++it)
			// cout << "," << *it;
	}
	cout << "}";
	cout << endl;
}

void printListUsingIterator(const list<string>& s) {
   	// TODO: Print a list of string using iterator, use "," to separate each object
   	cout << "{";
	if (!s.empty())
	{
		// cout << *s.begin();
		copy(s.begin(), s.end(), ostream_iterator<string>(cout, ","));
		// for (auto it = std::next(s.begin()); it != s.end(); ++it)
			// cout << "," << *it;
	}
	cout << "}";
	cout << endl;
}

void printMapUsingIterator(const map<string,int>& s) {
   	// TODO: Print a map using iterator, use '\t' to separate "key: " and "Value: "
	if (!s.empty())
	{
		for (auto it = s.begin(); it != s.end(); ++it)
			cout << "key: " << it->first << "\tValue: " << it->second << endl;
	}
}


//TODO: You may need to define a comparator function yourself here (for the sorting task)

int main() {

	cout << endl;
	cout << "************************** Part1: set **************************";
	cout << endl;

	set<string> Fictions, Movies;
	Fictions.insert("The Time Machine");
	Fictions.insert("Harry Potter");
	Fictions.insert("The Lord of the Rings");

	Movies.insert("The Shawshank Redemption");
	Movies.insert("City of God");
	Movies.insert("The Lord of the Rings");
	Movies.insert("Harry Potter");

	cout << "Set Fictions Content = " ;
	printSetUsingIterator(Fictions);
	cout << "Set Movies Content = " ;
	printSetUsingIterator(Movies);

    // Part 1 TODO: Complete the set operations: intersection
	set<string> interSet;

	set_intersection(
		Fictions.begin(), Fictions.end(),
		Movies.begin(), Movies.end(),
		std::inserter(interSet, interSet.end())
	);

   	cout << "Fictions intersect Movies Content = ";
   	printSetUsingIterator(interSet);

   	cout << endl;
	cout << "************************** Part2: list **************************";
	cout << endl;

	// Merge Fictions and Movies to listR
	list<string> listR;
   
   	// Part 2 TODO: Merge Fictions and Movies to listR
   	merge(
		Fictions.begin(), Fictions.end(),
		Movies.begin(), Movies.end(),
		std::inserter(listR, listR.end())
	);

	cout << "List R Content = ";
	printListUsingIterator(listR);

	// Part 2 TODO: 
    //Add a new string "Saw" at the end of the list
    //Add a new string "Avenger" at the head of the list
    // ADD YOUR CODE HERE
	listR.push_back("Saw");
	listR.push_front("Avenger");

	cout << "New R Content = ";
	printListUsingIterator(listR);

	 // Sort listR by movie name length ascendingly
    // ADD YOUR CODE HERE
	listR.sort([](const std::string& a, const std::string& b) { return a.length() < b.length(); });

	cout << "Sorted R Content = ";
	printListUsingIterator(listR);

   	cout << endl;
	cout << "************************** Part3: map **************************";
   	cout << endl;

    //map
	map<string,int> mapMovie;
	mapMovie.insert(make_pair("The Shawshank Redemption",1994));
	mapMovie.insert(make_pair("City of God",2002));
	mapMovie.insert(make_pair("The Lord of the Rings", 2002));
	mapMovie.insert(make_pair("Star Wars", 1977));
	mapMovie.insert(make_pair("Forest Gump", 1994));
	cout << "mapMovie Content: "<<endl;
	printMapUsingIterator(mapMovie);

    //Part3 TODO : Complete element search and deletion in mapMovie here
    // search "Star Wars" in map
	{
		map<string, int>::iterator it = mapMovie.find("Star Wars");
		if (it != mapMovie.end())
			cout << "Key found, the value is " << it->second << endl;
	}

    // delete "City of God" in map
	{
		map<string, int>::iterator it = mapMovie.find("City of God");
		if (it != mapMovie.end())
			mapMovie.erase(it);
	}

	cout << "mapMovie Content after deletion: "<<endl;
	printMapUsingIterator(mapMovie);

	return 0;
}
