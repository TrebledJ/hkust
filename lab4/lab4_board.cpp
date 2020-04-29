//
//  lab4_board.cpp
//  Copyright © 2020 TrebledJ. All rights reserved.
//

#include "lab4_board.hpp"

#include <iomanip>
#include <iostream>
#include <queue>


Board::Board(int size, int goal) : size{size}, goal{goal} {
    srand(uint16_t(time(0)));
    startNewGame();
}

void Board::startNewGame() {
    board = Grid(size, GridRow(size, 0));
    score = 0;
    generateNewTile();
}

bool Board::move(Direction dir) {
    bool change = false;
    switch (dir) {
        case UP: {
            for (int x = 0; x < size; ++x)
                change |= moveColumnUp(x);
        } break;
        case LEFT: {
            for (int y = 0; y < size; ++y)
                change |= moveRowLeft(y);
        } break;
        case DOWN: {
            for (int x = 0; x < size; ++x)
                change |= moveColumnDown(x);
        } break;
        case RIGHT: {
            for (int y = 0; y < size; ++y)
                change |= moveRowRight(y);
        } break;
        default:
            break;
    }
    
    if (change)
        generateNewTile();
    
    return change;
}

void Board::print() const {
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            cout << setw(5) << getTile({x, y});
        }
        cout << endl;
    }
    cout << "Score: " << score << endl;
    cout << endl;
}

int Board::getScore() const {
    return score;
}

bool Board::win() const {
    return maxTile() >= goal;
}

bool Board::lost() const {
    for (int y = 0; y < size; ++y)
        for (int x = 0; x < size; ++x)
            if (isTileEmpty({x, y})) {
                return false;
            }
    
    //  check adjacent cells
    for (int y = 0; y < size; ++y)
        for (int x = 0; x < size; ++x) {
            int cur = getTile({x, y});
            if (x > 0 && cur == getTile({x-1, y}))
                return false;
            if (x < size-1 && cur == getTile({x+1, y}))
                return false;
            if (y > 0 && cur == getTile({x, y-1}))
                return false;
            if (y < size-1 && cur == getTile({x, y+1}))
                return false;
        }
    
    return true;
}

void Board::setBoard(Board::Grid grid) {
    board = grid;
}

void Board::generateNewTile() {
    vector<Point> emptyTiles;
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            if (isTileEmpty({x, y}))
                emptyTiles.push_back({x, y});
        }
    }
    
    int idx = rand() % emptyTiles.size();
    int value = ((rand() % 2) + 1) * 2;
    setTile(emptyTiles.at(idx), value);
    addScore(value);
}

void Board::moveTile(const Point& from, const Point& to) {
    setTile(to, getTile(from));
    clearTile(from);
}

bool Board::moveRowLeft(int row) {
    auto past_row = getRow(row);
    
    //  combine and push tiles with queue
    queue<int> q;
    for (int x = 0; x < size; ++x) {
        if (!isTileEmpty({x, row}))
            q.push(getTile({x, row}));
    }
    
    clearRow(row);
    
    for (int avail_x = 0; !q.empty();) {
        int cur = q.front();
        q.pop();
        
        //  join tiles
        if (!q.empty() && cur == q.front()) {
            q.pop();
            addScore(cur);
            cur *= 2;
        }
        setTile({avail_x, row}, cur);
        avail_x++;
    }
    
    return getRow(row) != past_row;
}

bool Board::moveRowRight(int row) {
    auto past_row = getRow(row);
    
    //  combine and push tiles with queue
    queue<int> q;
    for (int x = size - 1; x >= 0; --x) {
        if (!isTileEmpty({x, row}))
            q.push(getTile({x, row}));
    }
    
    clearRow(row);
    
    for (int avail_x = size - 1; !q.empty();) {
        int cur = q.front();
        q.pop();
        
        //  join tiles
        if (!q.empty() && cur == q.front()) {
            q.pop();
            addScore(cur);
            cur *= 2;
        }
        setTile({avail_x, row}, cur);
        avail_x--;
    }
    
    return getRow(row) != past_row;
}

bool Board::moveColumnUp(int col) {
    auto past_col = getColumn(col);
    
    //  combine and push tiles with queue
    queue<int> q;
    for (int y = 0; y < size; ++y) {
        if (!isTileEmpty({col, y}))
            q.push(getTile({col, y}));
    }
    
    clearColumn(col);
    
    for (int avail_y = 0; !q.empty();) {
        int cur = q.front();
        q.pop();
        
        //  join tiles
        if (!q.empty() && cur == q.front()) {
            q.pop();
            addScore(cur);
            cur *= 2;
        }
        setTile({col, avail_y}, cur);
        avail_y++;
    }
    
    return getColumn(col) != past_col;
}

bool Board::moveColumnDown(int col) {
    auto past_col = getColumn(col);
    
    //  combine and push tiles with queue
    queue<int> q;
    for (int y = size - 1; y >= 0; --y) {
        if (!isTileEmpty({col, y}))
            q.push(getTile({col, y}));
    }
    
    clearColumn(col);
    
    for (int avail_y = size - 1; !q.empty();) {
        int cur = q.front();
        q.pop();
        
        //  join tiles
        if (!q.empty() && cur == q.front()) {
            q.pop();
            addScore(cur);
            cur *= 2;
        }
        setTile({col, avail_y}, cur);
        avail_y--;
    }
    
    return getColumn(col) != past_col;
}

void Board::clearRow(int row) {
    for (int x = 0; x < size; ++x)
        clearTile({x, row});
}

void Board::clearColumn(int col) {
    for (int y = 0; y < size; ++y)
        clearTile({col, y});
}

Board::GridRow Board::getRow(int row) const {
    return board.at(row);
}

Board::GridRow Board::getColumn(int col) const {
    GridRow column;
    for (int y = 0; y < size; ++y)
        column.push_back(getTile({col, y}));
    return column;
}

int Board::getTile(const Point &p) const {
    return board.at(p.y).at(p.x);
}

void Board::setTile(const Point& p, int value) {
    board.at(p.y).at(p.x) = value;
}

void Board::clearTile(const Point& p) {
    setTile(p, 0);
}

bool Board::isTileEmpty(const Point& p) const {
    return getTile(p) == 0;
}

int Board::maxTile() const {
    int max = 0;
    for (int y = 0; y < size; ++y)
        for (int x = 0; x < size; ++x)
            if (auto tile = getTile({x, y}); tile > max)
                max = tile;
                
    return max;
}

void Board::addScore(int s) {
    score += s;
}
