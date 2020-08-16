//
//  lab7_board.hpp
//  Copyright © 2020 TrebledJ. All rights reserved.
//

#ifndef lab7_board_hpp
#define lab7_board_hpp


#include "lab_helpers.hpp"
#include <vector>
#include <tuple>
#include <set>


struct Point { int x, y; };

int index(Point p, int size);
Point unindex(int idx, int size);


/**
 @brief Encapsulates a Lights-Out board. Maximum size is 8.
 */
class Board
{
public:
    using state_t = unsigned long long;
    
private:
    unsigned size;
    bool use_solution;
    state_t state = 0;
    std::set<int> saved_solution;
    Printer printer;
    
public:
    Board(unsigned size, bool use_solution = false);
    
    void toggle(int x, int y);
    inline void new_game();
    bool win() const;
    void print();
    
private:
    inline bool in_range(int x, int y) const;
    state_t move(state_t state, int x, int y) const;
    inline void toggle_one(state_t& state, int x, int y) const;
    inline bool light_at(int x, int y) const;
    
    void print_solution();
    state_t generate_random_state() const;

    std::set<int> solve(state_t state) const;
    
    std::optional<std::set<int>> solve_dfs(state_t state, int index = 0) const;
};



#endif /* lab7_board_hpp */
