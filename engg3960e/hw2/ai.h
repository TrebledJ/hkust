#ifndef EVAL_H
#define EVAL_H

#include "game.h"
#include <stdint.h>


typedef enum {
    DIR_R, DIR_D, DIR_RD, DIR_LD,   //  Directions a row could take. (Right, Down, RightDown, LeftDown.)
} Direction;

/**
 * RowID: An id of a row of char (can be diagonal/vertical as well).
 * 
 * We'll encode rows as an int: | dir (2 bits) | row (4 bits) | col (4 bits) |.
 */
typedef uint16_t RowID;
#define DIR_FLAG 0x300
#define ROW_FLAG 0x0F0
#define COL_FLAG 0x00F

inline uint8_t row_from_id(RowID id)                                    { return (id & ROW_FLAG) >> 4; }
inline uint8_t col_from_id(RowID id)                                    { return id & COL_FLAG; }
inline Direction dir_from_id(RowID id)                                  { return (id & DIR_FLAG) >> 8; }
inline RowID encode_id(uint8_t row, uint8_t col, Direction dir)         { return (dir << 8) | (row << 4) | col; }

bool ai_mm(Game* game, int* move_x, int* move_y);
int32_t ai_eval(const Field* field);


#endif
