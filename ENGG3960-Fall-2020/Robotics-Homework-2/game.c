#include "game.h"
#include "ai.h"
#include <stdio.h>


#define wrap(...)       do { __VA_ARGS__ } while (0)
#define await(...)      wrap(printf(__VA_ARGS__); getchar();)
static char buffer[256];


bool get_coord(int* x, int* y)
{
    printf("Please input a coordinate (x y): ");
    if (!fgets(buffer, 256, stdin)) return false;
    if (sscanf(buffer, "%d %d", x, y) < 2) return false;
    return true;
}

bool get_int(int* i)
{
    printf("Please input a number: ");
    if (!fgets(buffer, 32, stdin)) return false;
    if (sscanf(buffer, "%d", i) < 1) return false;
    return true;
}

static bool game_check_win(Game* game);
static void game_player_move(Game* game);
static void game_computer_move(Game* game);


/// Public methods:
void game_init(Game* game)
{
    game->state = STATE_START;
    game->turn = PLAYER1_TURN;
    field_init(&game->field);

#ifdef DEBUG_GAME_FIELD
    const char* field[] = DEBUG_GAME_FIELD;

    for (int i = 0; i < HEIGHT; i++)
        for (int j = 0; j < WIDTH; j++)
            game->field.grid[j][i] = field[j][i];
#endif

#ifdef DEBUG_FIRST_TURN
    game->turn = DEBUG_FIRST_TURN;
#endif
}

void game_begin(Game* game)
{
#ifdef DEBUG_GAME_MODE
    game->mode = DEBUG_GAME_MODE;
#else
    printf("Choose a game mode:\n");
    printf("1. Player vs. Player\n");
    printf("2. Player vs. Computer\n\n");
    
    int opt;
    while (!get_int(&opt) || opt <= 0 || opt > 2)
        printf("Invalid input! ");
    printf("\n");
    
    game->mode = (opt == 1 ? GAME_PVP : GAME_PVC);
    printf("%s mode selected!\n\n", opt == 1 ? "PVP" : "PVC");

    await("Press ENTER to begin the game!\n");
#endif

    game->state = STATE_GAME;
    field_print(&game->field);
}

void game_turn(Game* game)
{
    int32_t e = ai_eval(&game->field);
    printf("Eval: %d\n", e);

    game_computer_move(game);

#ifdef DEBUG_NO_TURN_LOOP    
    game->state = STATE_END;
#else
    //  Check if a player won a game.
    if (game_check_win(game))
        return;

    printf("\n");
    if (game->mode == GAME_PVP)
    {
        game_player_move(game);
        game->turn = !game->turn;
    }
    else
    {
        game_player_move(game);

        if (game_check_win(game))
            return;

        game_prompt();
        game_computer_move(game);
    }
#endif
}

void game_prompt()
{
    await("Press ENTER to continue!\n");
}

char symbol(PlayerTurn turn)
{
    return turn == PLAYER1_TURN ? PLAYER1 : turn == PLAYER2_TURN ? PLAYER2 : EMPTY;
}

/// Private methods:
static bool game_check_win(Game* game)
{
    char winner = field_check_win(&game->field);
    if (winner != EMPTY)
    {
        if (winner == PLAYER1)
            printf("Player 1 wins!\n");
        else if (winner == PLAYER2)
        {
            if (game->mode == GAME_PVC)
                printf("Computer wins!\n");
            else
                printf("Player 2 wins!\n");
        }
        else
            printf("Player ? wins!\n");

        game->state = STATE_END;
        return true;
    }
    return false;
}

static void game_player_move(Game* game)
{
    if (game->mode == GAME_PVP)
        printf("Player %d's Turn (%c):\n", game->turn + 1, symbol(game->turn));
    else
        printf("Player's Turn (%c):\n", symbol(game->turn));
    printf("===================\n");

    int x, y;
    while (!get_coord(&x, &y) || !is_valid_coord(y-1, x-1) || field_get(&game->field, y-1, x-1) != EMPTY)
        printf("Invalid input! ");

    //  Update field.
    field_set(&game->field, y-1, x-1, symbol(game->turn));
    field_print(&game->field);
    if (game->mode == GAME_PVP)
        printf("Player %d placed an %c at (%d, %d).\n", game->turn + 1, symbol(game->turn), x, y);
    else
        printf("You placed an %c at (%d, %d).\n", symbol(game->turn), x, y);
}

static void game_computer_move(Game* game)
{
    char sym = symbol(COMPUTER_TURN);
    printf("Computer's Turn (%c):\n", sym);
    printf("===================\n");

    int row = -1, col = -1;
    if (!ai_mm(game, &row, &col))
    {
        printf("Computer move failed.\n");
        return;
    }
    if (row == -1 && col == -1)
    {
        printf("Computer move failed: row == -1 && col == -1\n");
        return;
    }
    field_set(&game->field, row, col, sym);
    field_print(&game->field);
    printf("Computer placed an %c at (%d, %d).\n", sym, col+1, row+1);
}
