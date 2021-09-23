CXX=g++
CXXFLAGS=-std=c++17 -I.
DEPS=%.h

all: chocolate.exe burger.exe betterBurger.exe combo.exe

chocolate.exe: chocolate.o mainChocolate.cpp
	$(CXX) -o $@ $(CXXFLAGS) chocolate.o mainChocolate.cpp

burger.exe: burger.o bread.o meat.o mainBurger.cpp
	$(CXX) -o $@ $(CXXFLAGS) burger.o bread.o meat.o mainBurger.cpp

betterBurger.exe: burger.o bread.o betterMeat.o mainBurger.cpp
	$(CXX) -o $@ $(CXXFLAGS) burger.o bread.o betterMeat.o mainBurger.cpp

combo.exe: burger.o bread.o meat.o chocolate.o muffin.o mainCombo.cpp
	$(CXX) -o $@ $(CXXFLAGS) burger.o bread.o meat.o chocolate.o muffin.o mainCombo.cpp

%.o: %.cpp $(DEPS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
# On Windows, use: del *.o *.exe
# On Linux or MacOS, use: rm -f *.o *.exe
	rm -f *.o *.exe

