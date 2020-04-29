//
//  lab4_board.hpp
//  Copyright © 2020 TrebledJ. All rights reserved.
//

#ifndef lab4_board_hpp
#define lab4_board_hpp

#include <random>
#include <vector>


using namespace std;


enum Direction { UP, LEFT, DOWN, RIGHT };
struct Point {
    int x, y;
    Point() : x{0}, y{0} {}
    Point(int x, int y) : x{x}, y{y} {}
};


/**
 @brief Encapsules a 2048 game
 */
class Board {
public:
    int size;
    int score;
    int goal;
    
private:
    using Grid = vector<vector<int>>;
    using GridRow = vector<int>;
    
    Grid board;
    
public:
    explicit Board(int size = 3, int goal = 32);
    
    void startNewGame();
    
    /**
     @param  dir UP/LEFT/DOWN/RIGHT
     @return true if the board changed, otherwise false
     */
    bool move(Direction dir);
    
    /**
     @brief Prints the board out
     */
    void print() const;
    
    int getScore() const;
    
    /**
     @return true if the largest tile is greater than or equal to the goal
     */
    bool win() const;
    
    /**
     @return true if there are no more moves, otherwise false
     */
    bool lost() const;

    void setBoard(Grid grid);
    
private:
    void generateNewTile();
    void moveTile(const Point& from, const Point& to);
    
    /**
     @return true if the row/column changed, otherwise false
     */
    bool moveRowLeft(int row);
    bool moveRowRight(int row);
    bool moveColumnUp(int col);
    bool moveColumnDown(int col);
    
    void clearRow(int row);
    void clearColumn(int column);
    
    GridRow getRow(int row) const;
    GridRow getColumn(int col) const;
    
    int getTile(const Point& p) const;
    void setTile(const Point& p, int value);
    void clearTile(const Point& p);
    bool isTileEmpty(const Point &p) const;
    
    int maxTile() const;
    void addScore(int s);
};


#endif /* lab4_board_hpp */
