#ifndef ENGINE_H_
#define ENGINE_H_
#include <ostream>


#define X_COLOR\
	X(Black)\
	X(White)\
	X(Silver)\
	X(Grey)\
	X(Red)\
	X(Blue)\

enum Color
{
#define X(A) A,
X_COLOR
#undef X
};

std::ostream& operator<< (std::ostream& os, Color c);


class Engine
{
public:
	Engine(int nc);		// Conversion Constructor

	void Start();	//Start the engine
	void Stop();		//Stop the engine
	int getNumCylinder() const;

	virtual void print() const = 0;
	
private:
	int m_numCylinder;		//number of cylinders in the engine

};


#endif /* ENGINE_H_ */
