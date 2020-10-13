OBJ_DIR := build
# Note: the following is just an option to make sure Make won't automatically delete any object files after every build; don't worry about that
.PRECIOUS: $(OBJ_DIR)/%.o 


all: $(OBJ_DIR)/test1 $(OBJ_DIR)/test2 $(OBJ_DIR)/test3 $(OBJ_DIR)/test4 $(OBJ_DIR)/test5 $(OBJ_DIR)/test6 $(OBJ_DIR)/test7 $(OBJ_DIR)/test8 $(OBJ_DIR)/test9 $(OBJ_DIR)/test10

$(OBJ_DIR)/test%: $(OBJ_DIR)/main%.o $(OBJ_DIR)/Inventory.o $(OBJ_DIR)/Item.o
	g++ -std=c++11 -o $@ $^ 
# Note: $^ is not covered in lab1. It basically denotes all depended files, i.e., all 3 files "main%.o Inventory.o Item.o"

$(OBJ_DIR)/Item.o: Item.cpp Item.h
	g++ -std=c++11 -c $< -o $@ 

$(OBJ_DIR)/Inventory.o: Inventory.cpp Inventory.h
	g++ -std=c++11 -c $< -o $@ 

$(OBJ_DIR)/%.o: %.cpp
	g++ -std=c++11 -c $< -o $@ 
# Note: this meta rule (see lab 1) is to generate rules for main1.o, ..., main10.o 

clean:
	rm -f $(OBJ_DIR)/*.o $(OBJ_DIR)/*.exe
# On most Windows, use: del *.o *.exe
# On Linux or MacOS or some Windows (just try which command works for you), use: rm -f *.o *.exe