SRC_DIR := .
OBJ_DIR := build
SRC_FILES := $(wildcard $(SRC_DIR)/*.cpp)
OBJ_FILES := $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(SRC_FILES))
CXX=g++
CXXFLAGS := -std=c++11 -Wall -O3 -I.

partA: $(OBJ_DIR)/partA.exe
partB: $(OBJ_DIR)/partB.exe

$(OBJ_DIR)/partA.exe: $(OBJ_DIR)/mainA.o
	g++ $(LDFLAGS) $(CXXFLAGS) -o $@ $^

$(OBJ_DIR)/partB.exe: $(OBJ_DIR)/mainB.o
	g++ $(LDFLAGS) $(CXXFLAGS) -o $@ $^

$(OBJ_DIR)/mainA.o: $(SRC_DIR)/mainA.cpp $(SRC_DIR)/set.h
	g++ $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

$(OBJ_DIR)/mainB.o: $(SRC_DIR)/mainB.cpp
	g++ $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f *.o *.exe build/*.o build/*.exe