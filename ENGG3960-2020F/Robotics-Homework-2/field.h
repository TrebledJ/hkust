#ifndef FIELD_H
#define FIELD_H

#include <stdbool.h>


#define WIDTH 15
#define HEIGHT 15
#define EMPTY '.'
#define PLAYER1 'X'
#define PLAYER2 'O'


typedef struct {
    char grid[WIDTH][HEIGHT];
    int filled;
} Field;

/// modifiers:
void field_init(Field* field);
/**
 * @brief   Sets an element of the field to the given character.
 * @return  Whether the operation was successful.
 */
void field_set(Field* field, int row, int col, char c);
void field_unset(Field* field, int row, int col);

/// accessors:
/**
 * @brief   Returns the PLAYER1/PLAYER2/EMPTY.
 */
bool field_is_full(const Field* field);
char field_check_win(const Field* field);
void field_print(const Field* field);
char field_get(const Field* field, int row, int col);

/// static methods:
bool is_valid_coord(int x, int y);


#endif