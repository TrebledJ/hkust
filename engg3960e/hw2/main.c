#include "field.h"
#include "game.h"
#include <stdio.h>


int main()
{
    Game game;
    game_init(&game);
    game_begin(&game);

    // printf("State: %d\n", game.state);
    while (game.state == STATE_GAME)
    {
        game_turn(&game);
    }
}
