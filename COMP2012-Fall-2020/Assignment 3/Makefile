SRC_DIR := .
OBJ_DIR := build
ALL_SRC_FILES := $(wildcard $(SRC_DIR)/*.cpp)
MAIN_FILES := $(wildcard $(SRC_DIR)/main*.cpp)
SRC_FILES := $(filter-out $(MAIN_FILES), $(ALL_SRC_FILES))
OBJ_FILES := $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(SRC_FILES))
CXX=g++
CXXFLAGS := -std=c++11 -Wall -O3 -I. -fsanitize=address,undefined

DEPS := $(addprefix $(SRC_DIR)/,BST.h BSTImplementation.h BSTNode.h HashTable.h HashTableImplementation.h)

all: $(patsubst $(SRC_DIR)/main%.cpp, $(OBJ_DIR)/main%.exe, $(MAIN_FILES))

# Links .o files together into an executable.
$(OBJ_DIR)/main%.exe: $(OBJ_DIR)/main%.o $(OBJ_FILES) | $(OBJ_DIR)
	g++ $(LDFLAGS) $(CXXFLAGS) -o $@ $^

# Compile main file.
$(OBJ_DIR)/main%.o: $(SRC_DIR)/main%.cpp $(DEPS)
	g++ $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

# Compile related .o and .h files.
$(OBJ_DIR)/%.o: %.cpp %.h
	g++ $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

# Prepare the build dir.
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# Deletes build files (.o, .exe).
clean:
	rm -f *.o *.exe $(OBJ_DIR)/*.o $(OBJ_DIR)/*.exe