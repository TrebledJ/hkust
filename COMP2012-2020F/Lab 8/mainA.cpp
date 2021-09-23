#include <iostream>
#include <string>
#include "set.h"

using namespace std;

ostream& operator<<(ostream& os, pair<string, int> value) {
	os << "\n(" << value.first << " : " << value.second << ")";
	return os;
}

int main() {
	cout << "==========================================\n"
			<< "||             Movie List              ||\n"
			<< "==========================================\n" << endl;
	string movie1[3] = { "The Time Machine", "Harry Potter", "The Lord of the Rings" };
	string movie2[4] = { "The Shawshank Redemption", "City of God", "The Lord of the Rings", "Harry Potter" };
	string movie3[3] = { "Harry Potter", "The Lord of the Rings", "The Time Machine" };
	set<string> A(movie1, 3);
	set<string> B(movie2, 4);
	set<string> D(movie3, 3);

	cout << "1. Alice, Bob, and Dan are sharing their Movie Lists." << endl;

	cout << "Alice's Movie List (A) : ";
	cout << A << endl;
	cout << "|A| = " << A.cardinality() << endl << endl;

	cout << "Bob's Movie List (B) : ";
	cout << B << endl;
	cout << "|B| = " << B.cardinality() << endl << endl;

	cout << "Dan's Movie List (D) : ";
	cout << D << endl;
	cout << "|D| = " << D.cardinality() << endl << endl;

	set<string> C = A + B;
	cout
			<< "2. Carol collects the Movie List from Alice and Bob. (A union B)"
			<< endl;
	cout << "Carol's Movie List (C) : ";
	cout << C << endl;
	cout << "|C| = " << C.cardinality() << endl << endl;

	set<string> E = A * B;
	cout
			<< "3. Eve wants to find the movie(s) that would work for both Alice and Bob. (A intersect B)"
			<< endl;
	cout << "Eve's Movie List (E) : ";
	cout << E << endl;
	cout << "|E| = " << E.cardinality() << endl << endl;

	cout
			<< "4. Alice, Bob, and Dan want to find out if their Movie Lists share the same movie(s)."
			<< endl;
	cout << "A = ";
	cout << A << endl;
	cout << "B = ";
	cout << B << endl;
	cout << "D = ";
	cout << D << endl;
	cout << "A == B? " << ((A == B) ? "true" : "false") << endl;
	cout << "A == D? " << ((A == D) ? "true" : "false") << endl;

	cout << endl;

	cout << "==========================================\n"
			<< "||           Movie  Info           ||\n"
			<< "==========================================\n" << endl;
	set<pair<string, int>> directory;

	pair<string, int> info1 { "The Lord of the Rings", 2002 };
	pair<string, int> info2 { "City of God",2002 };
	pair<string, int> info3 { "The Shawshank Redemption",1994 };
	pair<string, int> info4 { "Star Wars", 1977 };
	pair<string, int> info5 { "Forest Gump", 1994};
	

	cout << "5. The initial state of directory : " << endl;
	cout << directory << endl << endl;
	cout << "6. Adding some movie info : " << endl;
	directory.addElement(info1);
	directory.addElement(info2);

	cout << directory << endl << endl;

	cout
			<< "7. Several students have suggested adding some old films : "
			<< endl;
	directory.addElement(info3);
	directory.addElement(info3);
	directory.addElement(info4);
	directory.addElement(info5);
	cout << directory << endl << endl;

	cout << "8. Oops, a student miswrote the release year of the film, let's remove that : "
			<< endl;
	directory.removeElement(info3);
	cout << directory << endl;

	return 0;
}
