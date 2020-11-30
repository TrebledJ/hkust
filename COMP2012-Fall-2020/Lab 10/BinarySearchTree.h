//
// Binary Search Tree Class
//

#ifndef BINARYSEARCHTREE_H
#define BINARYSEARCHTREE_H

#include <functional>
#include <iostream>
#include <iomanip>
#include <tuple>


template<typename T1, typename T2>
class BinarySearchTree {
private:
    class BinaryNode;

public:
    BinarySearchTree()
        : root(nullptr) 
    {

    }

    // (Deep) Copy constructor
    BinarySearchTree(const BinarySearchTree& src) 
        : root(src.clone(src.root))
    {
    }

    ~BinarySearchTree()
    {
        make_empty();
    };

    bool isEmpty() const
    {
        return !root;
    }

    bool contains(const T1& x, const T2& y) const
    {
        return contains(x, y, root);
    }

    void print_max() const;

    void print_tree() const
    {
        print_tree(root, 0);
    }

    void make_empty()
    {
        make_empty(root);
    }

    void insert(const T1& x, const T2& y)
    {
        insert(x, y, root);
    }

private:
    class BinaryNode
    {
    public:
        T1 x;
        T2 y;
        BinaryNode *left;
        BinaryNode *right;

        BinaryNode() :
                left(NULL), right(NULL) {}

        BinaryNode(const T1 &x, const T2 &y, BinaryNode* lt = NULL, BinaryNode* rt = NULL)
                : x(x), y(y), left(lt), right(rt) {}

        bool operator< (const BinaryNode& rhs) const { return std::make_tuple(x, y) < std::make_tuple(rhs.x, rhs.y); }
        bool operator== (const std::tuple<T1, T2>& rhs) const { return std::make_tuple(x, y) == rhs; }

        friend bool operator< (const BinaryNode& lhs, const std::tuple<T1, T2>& rhs) { return std::make_tuple(lhs.x, lhs.y) < rhs; }
        friend bool operator< (const std::tuple<T1, T2>& lhs, const BinaryNode& rhs) { return lhs < std::make_tuple(rhs.x, rhs.y); }
        friend std::ostream& operator<< (std::ostream& os, const BinaryNode& rhs)
        {
            return os << "(" << rhs.x << "," << rhs.y << ")";
        }
    };

private:
    BinaryNode *root;

    void insert(const T1& x, const T2& y, BinaryNode*& t);
    bool contains(const T1& x, const T2& y, BinaryNode* t) const;
    void make_empty(BinaryNode *t);
    void print_tree(BinaryNode *t, int depth) const;

    template<typename Acc>
    using Accumulator = std::function<Acc(const BinaryNode*, const Acc)>;

    //  Implementing these just for fun.
    template<typename Acc> Acc fold(Accumulator<Acc> f, Acc init) const;
    template<typename Acc> Acc fold_impl(const Accumulator<Acc>& f, const Acc& acc, const BinaryNode* node) const;

    BinaryNode *clone(BinaryNode* t) const
    {
        return !t ? NULL :
               new BinaryNode(t->x, t->y, clone(t->left), clone(t->right));
    };
};


// Print the BST out, the output is rotated -90 degrees.
template<typename T1, typename T2>
void BinarySearchTree<T1, T2>::print_tree(typename BinarySearchTree<T1, T2>::BinaryNode* t, int depth) const
{
    if (t == NULL)
        return;
    const int offset = 6;
    print_tree(t->right, depth + 1);
    std::cout << std::setw(depth * offset);
    std::cout << *t << std::endl;
    print_tree(t->left, depth + 1);
}


template<typename T1, typename T2>
void BinarySearchTree<T1, T2>::insert(const T1& x, const T2& y, BinarySearchTree<T1, T2>::BinaryNode*& node)
{
    // Insert a pair of key to the BST. If the entry already exists, you don't need to insert it again
    // i.e., there are no duplicate nodes.
    auto t = std::make_tuple(x, y);
    if (!node)
    {
        // Empty leaf node, means node hasn't been found.
        // So let's make a new one.
        node = new BinaryNode(x, y);
        return;
    }
    
    if (*node == t)
        return; //  Node already exists.

    insert(x, y, t < *node ? node->left : node->right);  // Recurse.
}


template<typename T1, typename T2>
bool BinarySearchTree<T1, T2>::contains(const T1& x, const T2& y, BinarySearchTree<T1, T2>::BinaryNode* node) const
{
    // Check if the BST contains the value (x,y).
    return node && (*node == std::make_tuple(x, y) || contains(x, y, node->left) || contains(x, y, node->right));
}


template<typename T1, typename T2>
void BinarySearchTree<T1, T2>::make_empty(BinarySearchTree<T1, T2>::BinaryNode* t)
{
     // Delete all nodes in the BST object.
     if (!t) return;

     make_empty(t->left);
     make_empty(t->right);
     delete t;
     t = nullptr;
}

// Print maximum key
template<typename T1, typename T2>
void BinarySearchTree<T1, T2>::print_max() const
{
    // Print max key. If the tree is empty, you should print "The maximum key is undefined." 
    if (!root)
    {
        std::cout << "The maximum key is undefined." << std::endl;
        return;
    }

    auto max = fold<const BinaryNode*>([](const BinaryNode* node, const BinaryNode* max)
                {
                    return node && *max < *node ? node : max;
                }, root);

    if (!max)
        throw std::runtime_error{"expected valid max node, got nullptr instead"};

    std::cout << "The maximum key is " << *max << std::endl;
}

template<typename T1, typename T2>
template<typename Acc>
Acc BinarySearchTree<T1, T2>::fold(BinarySearchTree<T1, T2>::Accumulator<Acc> f, Acc init) const
{
    return fold_impl(f, init, root);
}

template<typename T1, typename T2>
template<typename Acc>
Acc BinarySearchTree<T1, T2>::fold_impl(const BinarySearchTree<T1, T2>::Accumulator<Acc>& f, const Acc& acc, const BinarySearchTree<T1, T2>::BinaryNode* node) const
{
    if (!node) return acc;
    return fold_impl(f, fold_impl(f, f(node, acc), node->left), node->right);
}


#endif //BINARYSEARCHTREE_H
