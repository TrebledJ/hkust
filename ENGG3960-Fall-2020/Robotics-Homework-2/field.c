#include "field.h"
#include <stdio.h>

#define loop_all()  loop_wh(WIDTH, HEIGHT)
#define loop_wh(W, H) \
    for (int i = 0; i < (H); i++)\
        for (int j = 0; j < (W); j++)

#define check_line(r, c) \
    if (field->grid[row][col] == EMPTY) return false;\
    for (int i = 0; i < 5; i++)\
        if (field->grid[row][col] != field->grid[row + (r)][col + (c)])\
            return false;\
    return true;

static bool check_horizontal(const Field* field, int row, int col) { check_line(0, i); }
static bool check_vertical(const Field* field, int row, int col) { check_line(i, 0); }
static bool check_diagonal1(const Field* field, int row, int col)
{
    if (!is_valid_coord(row + 4, col + 4)) return false;
    check_line(i, i);
}
static bool check_diagonal2(const Field* field, int row, int col)
{
    if (!is_valid_coord(row + 4, col - 4)) return false;
    check_line(i, -i);
}


void field_init(Field* field)
{
    for (int i = 0; i < HEIGHT; ++i)
        for (int j = 0; j < WIDTH; ++j)
            field->grid[i][j] = EMPTY;
    field->filled = 0;
}

void field_set(Field* field, int row, int col, char c)
{
    if (field->grid[row][col] == EMPTY)
    {
        field->grid[row][col] = c;
        field->filled++;
    }
}

void field_unset(Field* field, int row, int col)
{
    if (field->grid[row][col] != EMPTY)
    {
        field->grid[row][col] = EMPTY;
        field->filled--;
    }
}

bool field_is_full(const Field* field)
{
    return field->filled == WIDTH * HEIGHT;
}

char field_check_win(const Field* field)
{
    //  Check horizontal.
    loop_wh(WIDTH - 5, HEIGHT) if (check_horizontal(field, i, j)) return field->grid[i][j];
    
    //  Check vertical.
    loop_wh(WIDTH, HEIGHT - 5) if (check_vertical(field, i, j)) return field->grid[i][j];

    //  Check diagonals.
    loop_all()
    {
        if (check_diagonal1(field, i, j)) return field->grid[i][j];
        if (check_diagonal2(field, i, j)) return field->grid[i][j];
    }

    return EMPTY;
}

void field_print(const Field* field)
{
    printf("  ");
    for (int j = 0; j < WIDTH; ++j)
        printf(" %d", (j + 1) % 10);
    printf(" x\n");

    for (int i = 0; i < HEIGHT; ++i)
    {
        printf("%d ", (i+1) % 10);
        for (int j = 0; j < WIDTH; ++j)
            printf(" %c", field->grid[i][j]);
        printf("\n");
    }
    printf("y\n");
}

char field_get(const Field* field, int row, int col)
{
    return field->grid[row][col];
}

bool is_valid_coord(int row, int col)
{
    return 0 <= row && row < HEIGHT && 0 <= col && col < WIDTH;
}
