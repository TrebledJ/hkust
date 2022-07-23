SRC_DIR := .
OBJ_DIR := build
SRC_FILES := $(wildcard $(SRC_DIR)/*.c)
OBJ_FILES := $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRC_FILES))
CC=gcc
CFLAGS := -std=c18 -Wall -O3 -I.
DEPS=$(wildcard $(SRC_DIR)/*.h)

all: $(OBJ_DIR)/main.exe

$(OBJ_DIR)/main.exe: $(OBJ_FILES)
	$(CC) $(LDFLAGS) $(CXXFLAGS) -o $@ $^

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c $(SRC_DIR)/%.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $<

clean:
	rm -f *.o *.exe $(OBJ_DIR)/*.o $(OBJ_DIR)/*.exe