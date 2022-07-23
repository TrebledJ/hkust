//
//  lab7_board.cpp
//  Copyright © 2020 TrebledJ. All rights reserved.
//

#include "lab7_board.hpp"

#include <iostream>
#include <algorithm>
#include <ctime>
#include <vector>
#include <queue>
#include <unordered_set>


/**
 Static functions and variables.
 */
Printer debug;
template<class T>
std::set<T> operator| (std::set<T> lhs, std::set<T> const& rhs)
{
    for (auto const& e : rhs)
        lhs.insert(e);
    return lhs;
}

template<class T>
std::set<T> operator- (std::set<T> const& lhs, std::set<T> const& rhs)
{
    std::set<T> exclusion = lhs;
    for (auto const& e : rhs)
        if (auto it = exclusion.find(e); it != exclusion.end())
            exclusion.erase(it);
    return exclusion;
}

template<class T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
std::string to_bin(T num)
{
    std::string result;
    while (num) {
        result = char((num & 1) + '0') + result;
        num >>= 1;
    }
    return result;
}


int index(Point p, unsigned size)
{
    return p.x * size + p.y;
}

Point unindex(int idx, unsigned size)
{
    return {int(idx / size), int(idx % size)};
}

/**
 Class member functions
 */
Board::Board(unsigned size, bool use_solution) : size{size}, use_solution{use_solution}, printer{std::cout}
{
    srand(uint16_t(time(0)));
    debug.stop();
    
    if (size > 8)
    {
        std::cout << "max board size exceeded (" << size << "), setting board size to 8" << std::endl;
        size = 8;
    }
        
    new_game();
}
        
void Board::toggle(int x, int y)
{
    if (!in_range(x, y))
        return;
    
    state = move(state, x, y);
    if (auto it = saved_solution.find(index({x, y}, size)); it != saved_solution.end())
        saved_solution.erase(it);
    else
        saved_solution.clear();
}
        
void Board::new_game()
{
    state = 0;
    for (auto i = 0; i < size*size; ++i)
        if (rand() % 2) {
            auto [x, y] = unindex(i, size);
            state = move(state, x, y);
            debug("applied xy", x, y);
        }
        
    //  add debugger states here
    //  state = 0b010111010;
}
    
bool Board::win() const
{
    return state == 0;
}
        
void Board::print() {
    const int gap = 3;
    std::cout.fill(' ');
    printer.printw(gap, ' ');
    printer.printw(gap, ' ');
    for (int i = 0; i < size; ++i) {
        printer.printw(gap, i);
    }
    printer.print();
    printer.printw(gap, ' ');
    printer.printw(gap, '+');
    for (int i = 0; i < size; ++i) printer.printw(gap, '-');
    printer.print(" y");
    
    for (int i = 0; i < size; ++i) {
        printer.printw(gap, i);
        printer.printw(gap, '|');
        for (int j = 0; j < size; ++j) {
            printer.printw(gap, light_at(i, j) == 0 ? '.' : 'O');
        }
        printer.print();
    }
    printer.printw(2*gap-1, ' ');
    printer.print('x');
    printer.print();
    
    if (use_solution)
        print_solution();
}
        
bool Board::in_range(int x, int y) const {
    return 0 <= x && x < size && 0 <= y && y < size;
}

Board::state_t Board::move(state_t state, int x, int y) const {
    toggle_one(state, x+0, y+0);
    toggle_one(state, x+1, y+0);
    toggle_one(state, x+0, y+1);
    toggle_one(state, x-1, y+0);
    toggle_one(state, x+0, y-1);
    return state;
}

void Board::toggle_one(state_t& state, int x, int y) const {
    if (in_range(x, y))
        state ^= state_t(1 << index({x, y}, size));
}

bool Board::light_at(int x, int y) const {
    return state & state_t(1 << index({x, y}, size));
}
        
void Board::print_solution()
{
    if (state == 0)
        return;
    
    if (saved_solution.empty()) {
        saved_solution = solve(state);
    }
    
    debug("solution has", saved_solution.size(), "steps");
    
    if (saved_solution.empty())
        printer.print("(no solution)");
    else {
        for (int m : saved_solution)
        {
            auto [x, y] = unindex(m, size);
            std::cout << " (" << x << "," << y << ")";
        }
        printer.print();
    }
}

Board::state_t Board::generate_random_state() const
{
    return rand() % state_t(1 << (size*size));
}

std::set<int> Board::solve(state_t state) const
{
    std::set<int> possible_moves;
    for (auto i = 0; i < size*size; ++i) possible_moves.insert(i);
    
    auto set = solve_dfs(state);
    return set.value_or(std::set<int>{});
}
    
std::optional<std::set<int>> Board::solve_dfs(state_t state, int index) const
{
    if (state == 0)
        return std::set<int>{};
    
    if (index == size*size)
        return std::nullopt;
    
    if (auto unactivated = solve_dfs(state, index + 1))
        return unactivated;
    
    auto [x, y] = unindex(index, size);
    if (auto activated = solve_dfs(move(state, x, y), index + 1))
        return (*activated) | std::set{index};
    
    return std::nullopt;
}
