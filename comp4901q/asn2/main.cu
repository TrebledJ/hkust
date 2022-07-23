#include "config.h"
#include "matrix.h"
#include "problems.h"
#include "utils.h"

using namespace Utils::Input;


int main(int argc, char** argv)
{
    int32_t row, col;
    input(
        "(<row> <col>) >>> ", [](int32_t row, int32_t col) { return row > 0 && col > 0; }, row, col);

    std::cout << std::endl;

    Context ctx(row, col);
    ctx.matrix.generate(RANDOM_LO, RANDOM_HI);
    ctx.num_runs = RUNS;
#ifdef PRINT_OUTPUT
    ctx.print_output = (col <= OUTPUT_COL_LIMIT || row <= OUTPUT_ROW_LIMIT);
#else
    ctx.print_output = false;
#endif

    std::cout << "\n";
    std::cout << "#runs: " << ctx.num_runs << "\n";
    std::cout << "#rows: " << row << "\n";
    std::cout << "#columns: " << col << "\n";
    std::cout << "matrix size: " << ctx.matrix.size() << "\n";

    if (ctx.print_output)
    {
        std::cout << "matrix:\n" << ctx.matrix;
    } else
    {
        std::cout << std::endl;
    }

#ifdef RUN_PROBLEM_1
    problem1(ctx);
#endif

#ifdef RUN_PROBLEM_2
    problem2(ctx);
#endif
}