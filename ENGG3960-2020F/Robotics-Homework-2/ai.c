#include "ai.h"
#include "field.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>


#define iter_r(row, col) for (int r = row, c = col; c < col + 5; c++)
#define iter_d(row, col) for (int r = row, c = col; r < row + 5; r++)
#define iter_rd(row, col) for (int r = row, c = col; r < row + 5; r++, c++)
#define iter_ld(row, col) for (int r = row, c = col; r < row + 5; r++, c--)

#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))

#define X_DIRECTION\
    X(DIR_R, iter_r)\
    X(DIR_D, iter_d)\
    X(DIR_RD, iter_rd)\
    X(DIR_LD, iter_ld)

static uint8_t ai_count_row(const Field* field, uint8_t row, uint8_t col, Direction dir, char match);
static int32_t ai_mm_dfs(Field* field, int* move_r, int* move_c, PlayerTurn player, int depth, int alpha, int beta);

/// Public methods:
bool ai_mm(Game* game, int* move_r, int* move_c)
{
    // printf("Beginning ai_mm()...\n");
    int best_score = ai_mm_dfs(&game->field, move_r, move_c, COMPUTER_TURN, 3, INT32_MIN, INT32_MAX);
    // printf("Finished ai_mm(). move=(%d, %d)\n", *move_c + 1, *move_r + 1);
    return best_score != INT32_MIN;
}

int32_t ai_eval(const Field* field)
{
    int32_t count_x_comp[6] = {0};
    int32_t count_x_player[6] = {0};
    int32_t weights[6] = {0, 1, 10, 100, 1000, 1000000};

    //  We'll find all the different combinations of 5 and count how much "progress" there is to fulfilling that 5.
    //  For performance, we'll only cache counts of 3+.

    //  TODO: if count >= 3, cache the id.
    //  TODO: include multi-spots in eval?
    for (int row = 0; row < HEIGHT; row++)
        for (int col = 0; col < WIDTH - 5; col++)
        {
            int count_comp = ai_count_row(field, row, col, DIR_R, symbol(COMPUTER_TURN));
            int count_player = ai_count_row(field, row, col, DIR_R, symbol(PLAYER1_TURN));
            count_x_comp[count_comp]++;
            count_x_player[count_player]++;
        }

    for (int row = 0; row < HEIGHT - 5; row++)
        for (int col = 0; col < WIDTH; col++)
        {
            int count_comp = ai_count_row(field, row, col, DIR_D, symbol(COMPUTER_TURN));
            int count_player = ai_count_row(field, row, col, DIR_D, symbol(PLAYER1_TURN));
            if (count_comp == 5)    return weights[5];
            if (count_player == 5)  return weights[5];
            count_x_comp[count_comp]++;
            count_x_player[count_player]++;
        }

    for (int row = 0; row < HEIGHT - 5; row++)
        for (int col = 0; col < WIDTH - 5; col++)
        {
            int count_comp = ai_count_row(field, row, col, DIR_RD, symbol(COMPUTER_TURN));
            int count_player = ai_count_row(field, row, col, DIR_RD, symbol(PLAYER1_TURN));
            count_x_comp[count_comp]++;
            count_x_player[count_player]++;
        }

    for (int row = 0; row < HEIGHT - 5; row++)
        for (int col = 4; col < WIDTH; col++)
        {
            int count_comp = ai_count_row(field, row, col, DIR_LD, symbol(COMPUTER_TURN));
            int count_player = ai_count_row(field, row, col, DIR_LD, symbol(PLAYER1_TURN));
            count_x_comp[count_comp]++;
            count_x_player[count_player]++;
        }
    
    // printf("=============\n");
    // printf("Eval summary:\n");
    // for (int i = 1; i < 6; i++)
    //     printf("count[%d]: %d / %d\n", i, count_x_comp[i], count_x_player[i]);

    //  Compute final sum.
    int32_t sum = 0;
    for (int i = 0; i < 6; i++)
        sum += (count_x_comp[i] - count_x_player[i]) * weights[i];
    return sum;
}

/// Private methods:
/**
 * @brief   Counts the progress of `match` to fulfilling 5-in-a-row at the given RowID (row, col, dir).
 *          If the row is obstructed by another non-EMPTY character, count is 0.
 * @param   field The field to check.
 * @param   row Row of the target cell.
 * @param   col Column of the target cell.
 * @param   dir Direction to check.
 * @param   match The char to match against.
 */
static uint8_t ai_count_row(const Field* field, uint8_t row, uint8_t col, Direction dir, char match)
{
#define X(enum, iter_x)                          \
    case enum:                                   \
        iter_x(row, col)                         \
        {                                        \
            if (field->grid[r][c] == match)      \
                count++;                         \
            else if (field->grid[r][c] != EMPTY) \
                return 0;                        \
        }                                        \
        break;

    uint8_t count = 0;
    switch (dir)
    {
    X_DIRECTION
    }
    return count;

#undef X
}

static int32_t ai_mm_dfs(Field* field, int* move_r, int* move_c, PlayerTurn player, int depth, int alpha, int beta)
{
    if (depth == 0 || field_is_full(field))
        return (depth + 1) * ai_eval(field);  //  Eval. The closer the node, the more significant the score.

    int32_t best_score = (player == COMPUTER_TURN ? INT32_MIN : INT32_MAX);

    for (int row = 0; row < HEIGHT; row++)
        for (int col = 0; col < WIDTH; col++)
        {
            if (field_get(field, row, col) != EMPTY) //  Occupied.
                continue;

            //  Simulate a move in a cell. We'll undo the move later.
            field_set(field, row, col, symbol(player));

            int32_t score = ai_mm_dfs(field, NULL, NULL, !player, depth-1, alpha, beta);
            if (col == 10)
            {
                printf("%*sscore (%d, %d): %d\n", 4*(3-depth), " ", col+1, row+1, score);
            }
            if (player == COMPUTER_TURN)
            {
                //  Max player.
                if (score > best_score)
                {
                    if (beta < score)
                    {
                        field_unset(field, row, col);
                        return score;
                    }

                    //  Update move.
                    if (move_r && move_c)
                    {
                        printf("Score: %d <> %d\n", score, best_score);
                        printf("Update computer move to (%d, %d)\n", col+1, row+1);
                        *move_r = row;
                        *move_c = col;
                    }

                    //  Update score.
                    alpha = best_score = score;
                }
            }
            else
            {
                //  Min player.
                if (score < best_score)
                {
                    if (score <= alpha)
                    {
                        field_unset(field, row, col);
                        return score;
                    }

                    //  Update score.
                    beta = best_score = score;
                }
            }
            
            field_unset(field, row, col);
        }

    return best_score;
}