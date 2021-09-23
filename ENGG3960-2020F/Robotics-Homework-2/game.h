#ifndef GAME_H
#define GAME_H

#include "field.h"

#if 1
//  Debug Mode.
#define DEBUG_GAME_FIELD \
   {"...............",\
    "...............",\
    "...X...........",\
    "...X......O....",\
    "..........O....",\
    "...X......O....",\
    "...X......O....",\
    "...............",\
    "...............",\
    "...............",\
    "...............",\
    "...............",\
    "...............",\
    "...............",\
    "...............",\
    }

#define DEBUG_FIRST_TURN    COMPUTER_TURN
#define DEBUG_GAME_MODE     GAME_PVC
#define DEBUG_NO_TURN_LOOP
#endif


typedef enum {
    PLAYER1_TURN,
    PLAYER2_TURN,
    COMPUTER_TURN = PLAYER2_TURN,
} PlayerTurn;

typedef enum {
    GAME_PVP,
    GAME_PVC,
} GameMode;

typedef enum {
    STATE_START,
    STATE_GAME,
    STATE_END,
} State;

typedef struct {
    Field field;
    State state;
    GameMode mode;
    PlayerTurn turn;
} Game;

void game_init(Game* game);
void game_begin(Game* game);
void game_turn(Game* game);
void game_prompt();
char symbol(PlayerTurn turn);


#endif