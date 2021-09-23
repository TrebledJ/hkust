OBJ_DIR = build

# Note: the following is just an option to make sure Make won't automatically delete any object files after every build; don't worry about that
.PRECIOUS: $(OBJ_DIR)/%.o 

all: test1 test2 test3

test%: $(OBJ_DIR)/main%.o $(OBJ_DIR)/ledger.o $(OBJ_DIR)/account.o $(OBJ_DIR)/transactions.o $(OBJ_DIR)/expenses.o $(OBJ_DIR)/income.o
	g++ -std=c++11 -o $@ $^ 
# Note: $^ is not covered in lab1. It basically denotes all depended files, i.e., all 6 files "main%.o ledger.o account.o transactions.o expenses.o income.o"

$(OBJ_DIR)/main%.o: main%.cpp
	g++ -std=c++11 -c $< -o $@ 

$(OBJ_DIR)/%.o: %.cpp %.h
	g++ -std=c++11 -c $< -o $@ 
# Note: this meta rule (see lab 1) is to generate rules for main1.o, ..., main3.o 

clean:
	rm -f $(OBJ_DIR)/*.o $(OBJ_DIR)/*.exe
	rm test1
	rm test2
	rm test3
# On most Windows, use: del *.o *.exe
# On Linux or MacOS or some Windows (just try which command works for you), use: rm -f *.o *.exe