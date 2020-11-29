OBJ_DIR = build

# Note: the following is just an option to make sure Make won't automatically delete any object files after every build; don't worry about that
.PRECIOUS: %.o 

all: test1 test2 test3 test5 test6 test7 test8 test9 test10 test12 test13 test14 test15 test16 test17 test18 test19 test20 test21 test22 test23 test24 test25 test27 test28 test29 test30 test31 test32 test33 test34 test35 test36 test37 test38 test39 test40 test41 test42 test43 test44 test45 test46 test48 test50

test%: $(OBJ_DIR)/main%.o $(OBJ_DIR)/ledger.o $(OBJ_DIR)/account.o $(OBJ_DIR)/transactions.o $(OBJ_DIR)/expenses.o $(OBJ_DIR)/income.o
	g++ -std=c++11 -o $@ $^ 

$(OBJ_DIR)/main%.o: main%.cpp
	g++ -std=c++11 -c $< -o $@ 

$(OBJ_DIR)/%.o: %.cpp %.h
	g++ -std=c++11 -c $< -o $@ 

# %.o: %.cpp
# 	g++ -std=c++11 -c $< -o $@ 

clean:
	rm -f $(OBJ_DIR)/*.o $(OBJ_DIR)/*.exe
	rm -f test*

cleanexec:
	rm -f test*