/*
* Engine.cpp
*/

#include <iostream>
#include "Engine.h"
using namespace std;


std::ostream& operator<< (std::ostream& os, Color c)
{
	#define X(A) #A,
	const char* s[] = {X_COLOR};
	return os << s[c];
	#undef X
}


Engine::Engine(int nc)
	: m_numCylinder{nc}
{
}

int Engine::getNumCylinder() const { return m_numCylinder; }

//Start the engine
void Engine::Start(){
	cout << getNumCylinder() << "-cylinder engine started." << endl;
}

//Stop the engine
void Engine::Stop(){
	cout << getNumCylinder() << "-cylinder engine stopped." << endl;
}

