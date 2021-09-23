//
// Operations on Binary Tree
//

#ifndef BTREEEXERCISES_H
#define BTREEEXERCISES_H

#include <iostream>
#include <queue>
#include "BtreeNode.h"


template<class T>
int tree_height(const BtreeNode<T> *root);

template<class T>
int count_nodes(const BtreeNode<T> *root);

template<class T>
BtreeNode<T> *mirror(const BtreeNode<T> *root);

template<class T>
bool is_complete(const BtreeNode<T> *root);


template <class T>
BtreeNode<T> *lowest_common_ancestor(BtreeNode<T>* root, const T& a, const T& b);

template <class T>
void test_lowest_common_ancestor_output(BtreeNode<T>* root, const T& a, const T& b);

template <class T>
bool is_bst(BtreeNode<T>* root);


namespace
{
	template<typename T>
	int tree_height_impl(const BtreeNode<T>* root)
	{
		if (!root) return 0;

		const BtreeNode<T>* left = root->get_left();
		const BtreeNode<T>* right = root->get_right();
		return 1 + std::max(tree_height_impl(left), tree_height_impl(right));
	}
}


// Tree Height -> Calculate the height of a binary tree
template<typename T>
int tree_height(const BtreeNode<T>* root)
{
	return root ? tree_height_impl(root) - 1 : 0;
}


// Count Nodes -> Calculate the number of nodes of a binary tree.
template<typename T>
int count_nodes(const BtreeNode<T>* root)
{
	if (!root) return 0;

	const BtreeNode<T>* left = root->get_left();
	const BtreeNode<T>* right = root->get_right();
	return 1 + count_nodes(left) + count_nodes(right);
}


// Mirror -> Create a "new" mirror tree
template<typename T>
BtreeNode<T>* mirror(const BtreeNode<T> *root)
{
	if (!root) return nullptr;

	const BtreeNode<T>* left = root->get_left();
	const BtreeNode<T>* right = root->get_right();

	return new BtreeNode<T>(root->get_data(), mirror(right), mirror(left));
}


namespace
{
	template<typename T>
	inline bool is_leaf(const BtreeNode<T>* node)
	{
		return (node ? !node->get_left() && !node->get_right() : true);
	}

	template<typename T>
	bool is_complete_impl(const BtreeNode<T>* root)
	{
		const BtreeNode<T>* left = root->get_left();
		const BtreeNode<T>* right = root->get_right();
		if (right && !left)
			return false;

		if (!is_leaf(left) && !is_leaf(right))
			return is_complete(left) && is_complete(right);

		if ((left && right && is_leaf(left) && is_leaf(right))
			|| (left && !right && is_leaf(left)))
		{
			return true;
		}

		return false;
	}
}


// Is Complete -> Check whether a binary tree is complete.
template<typename T>
bool is_complete(const BtreeNode<T>* root)
{
	if (!root) return true;

	const BtreeNode<T>* left = root->get_left();
	const BtreeNode<T>* right = root->get_right();

	//	For non-leaf nodes, check that the height is the same.
	if (!is_leaf(left) && !is_leaf(right) && tree_height(left) != tree_height(right))
		return false;

	if (right && !left)
		return false;

	return is_complete_impl(root);
}

// lowest_common_ancestor -> Find which is the lowest common ancestor of the two binary tree nodes. 
// The functions is_ancestor_of() may be helpful here.
template<typename T>
BtreeNode<T>* lowest_common_ancestor(BtreeNode<T>* root, const T& a, const T& b)
{
	BtreeNode<T>* left = root->get_left();
	BtreeNode<T>* right = root->get_right();
	T data = root->get_data();
	if (!left && !right) return nullptr;
	if ((data == a && root->is_ancestor_of(b)) || (data == b && root->is_ancestor_of(a))) return root;
	if (left && !right) return lowest_common_ancestor(left, a, b);
	if (right && !left) return lowest_common_ancestor(right, a, b);

	const bool la = left->is_ancestor_of(a);
	const bool ra = right->is_ancestor_of(a);
	const bool lb = left->is_ancestor_of(b);
	const bool rb = right->is_ancestor_of(b);
	if ((la && rb) || (lb && ra)) return root;
	if (la && lb) return lowest_common_ancestor(left, a, b);
	if (ra && rb) return lowest_common_ancestor(right, a, b);
	return nullptr;
}

namespace
{
	//	Determines if the subtree given by `root` is a BST by clamping the lower and upper bounds.
	template<typename T>
	bool is_bst_impl(BtreeNode<T>* root, const T& low, const T& high)
	{
		if (!root) return true;
		const T data = root->get_data();
		if (data < low || data > high)
			return false;
		
		return is_bst_impl(root->get_left(), low, data) && is_bst_impl(root->get_right(), data, high);
	}
}

// is_bst ->  Check whether a binary tree is also a binary search tree
// The functions find_min() and find_max() may be helpful here.
template<typename T>
bool is_bst(BtreeNode<T>* root)
{
	return is_bst_impl(root, root->find_min(), root->find_max());
}


template <class T>
void
test_lowest_common_ancestor_output(BtreeNode<T>* root, const T& a, const T& b)
{
	BtreeNode<T>* lca = lowest_common_ancestor(root, a, b);
	if (lca)
		std::cout << "Lowest common ancestor of " << a << " and " << b << " is " << lca->get_data() << std::endl;
	else
		std::cout << a << " and " << b << " have no common ancestor" << std::endl;
}


#endif //BTREEEXERCISES_H
