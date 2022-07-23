# do NOT modify this file
CC = gcc
CXX = g++
LEX = flex
YACC = bison
RM = rm

.PHONY: clean

SMILEScal: SMILEScal.c SMILEScal.tab.c helpers.c
	$(CC) -std=c99 -Wpedantic -Wno-newline-eof -o $@ $^

SMILEScal.c: SMILEScal.lex
	$(LEX) -o $@ $<

SMILEScal.tab.c: SMILEScal.y
	$(YACC) -d $<

clean: 
	$(RM) -rf SMILEScal.c SMILEScal SMILEScal.tab.* 
