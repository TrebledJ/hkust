SRC_DIR := .
OBJ_DIR := build
SRC_FILES := $(wildcard $(SRC_DIR)/*.cpp)
OBJ_FILES := $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(SRC_FILES))
CXX=g++
CXXFLAGS := -std=c++11 -Wall -O3 -I.

all: $(OBJ_DIR)/main.exe

$(OBJ_DIR)/main.exe: $(OBJ_FILES)
	g++ $(LDFLAGS) $(CXXFLAGS) -o $@ $^

$(OBJ_DIR)/main.o: $(SRC_DIR)/main.cpp $(SRC_DIR)/BTreeExercises.h $(SRC_DIR)/BinarySearchTree.h
	g++ $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f *.o *.exe build/*.o build/*.exe