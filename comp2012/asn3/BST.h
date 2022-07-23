
#ifndef ASG3_BST_H
#define ASG3_BST_H

#include <list>

using namespace std;

//BST is a Binary Search Tree class
//Key is always a string
//for comparing keys, simply use C++ string comparison
//e.g. "a > b" is the expression to check if string a is larger than string b
//keys are unique (to be made sure whenever we add a new item to the tree)
//Value to be stored in each node is of type T
//You may assume T supports proper operator= and copy construction (i.e. deep copy will be used if necessary)
template <typename T>
class BST
{
private:
public:
#include "BSTNode.h"            //since we only use BSTNode in BST class, we include it here
    BSTNode<T> *root = nullptr; //the root node; empty tree's root is nullptr

    //constructor
    BST() = default;

    //destructor
    //given; do not modify
    ~BST() { delete root; }

    //deep copy constructor
    BST(const BST &another);

    //return true if the tree is empty (i.e. no node at all)
    //return false otherwise
    bool isEmpty() const;

    //add a node to the tree, according to the given key and value
    //you have to use the exact algorithm described in the lecture notes
    //you should have the exact same result as our sample output
    //it should do nothing to the tree and return false when there is already a node that has the same key
    //otherwise, it should add the node and return true
    //again, assume T supports proper operator= and copy construction (i.e. deep copy will be used if necessary)
    bool add(string key, T value);

    //remove a node from the tree, according to the given key
    //if the key is found, it should remove the node and return true
    //otherwise, return false
    //note: you have to use the exact algorithm described in the lecture notes
    //when you need to delete a node that has 2 children,
    //you must replace the deleted node with the minimum node in its right sub-tree
    bool remove(string key);

    //return the pointer to the node's value (i.e., T*) given a key
    //if the key is not found in the tree, return nullptr
    T *get(string key) const;

    //return a list of object Ts which is inside the range
    //assume start <= end
    //key that between start (inclusive) <= key <= end (inclusive)
    //the order of the list of object should follow the ascending order of key
    //the smallest key is in the head, the largest key is in the tail
    //if there are no keys matched, just return an initialized list with size 0 (size 0).
    //!!!Don't need to modify this function, instead modify searchBetweenHelper
    list <T> *getBetweenRange(string start, string end) const {
        if (start > end) {
            string temp = end;
            end = start;
            start = temp;
        }
        list <T> *resultList = new list<T>;
        this->getBetweenRangeHelper(this, start, end, resultList);

        return resultList;
    }

    //implement it according to the getBetweenRange function instruction
    //Hint: tree traversal
    //Suggestion: use recursion method :)
    void getBetweenRangeHelper(const BST<T> *current_bst, string start, string end, list <T> *resultList) const;

    //return the subtree that has the minimum-value node as its root node
    //return nullptr if the tree is empty
    const BST *findMin() const;

    //print the tree to the ostream
    //note that displayed tree is rotated 90 degree clockwise
    //given; just like the rest of this file, do not modify
    void printTree(std::ostream &os, int depth = 0) const
    {
        if (isEmpty())
        {
            if (depth == 0)
                os << "Nothing to see here!" << std::endl;
            return;
        }

        root->right.printTree(os, depth + 1);

        for (int j = 0; j < depth; j++)
            os << "\t\t";
        os << "(" << root->key << ")" << std::endl;

        root->left.printTree(os, depth + 1);
    }
};

#include "BSTImplementation.h"

#endif //ASG3_BST_H
