SRC_DIR := .
OBJ_DIR := build
SRC_FILES := $(wildcard $(SRC_DIR)/*.cpp)
OBJ_FILES := $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(SRC_FILES))
CXX=g++
CXXFLAGS := -std=c++17 -Wall -O3 -I.

all: main.exe

main.exe: $(OBJ_FILES)
	g++ $(LDFLAGS) $(CXXFLAGS) -o $@ $^

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	g++ $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f *.o *.exe build/*.o build/*.exe