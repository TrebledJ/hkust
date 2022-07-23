#ifndef ASG3_BSTNODE_H
#define ASG3_BSTNODE_H


//BSTNode is the node struct that is used in the BST class
template <typename S>
struct BSTNode
{
    string key;
    S value;

    BST<S> left; // left sub-tree (left child)
    BST<S> right; // right sub-tree (right child)
    BSTNode(string key, S value) : key(key), value(value) {};
};


#endif //ASG3_BSTNODE_H
