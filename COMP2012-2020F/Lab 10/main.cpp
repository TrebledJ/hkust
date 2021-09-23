#include <iostream>
#include <vector>
#include "BtreeNode.h"
#include "BTreeExercises.h"
#include "BinarySearchTree.h"
#include "PrintBinaryTree.h"
using namespace std;
void
test_lowest_common_ancestor()
{

	BtreeNode<char>* nodeB = new BtreeNode<char>('B', new BtreeNode<char>('D', new BtreeNode<char>('H'), new BtreeNode<char>('I')), 
                                                      new BtreeNode<char>('E', new BtreeNode<char>('J'), new BtreeNode<char>('K')));
	BtreeNode<char>* nodeC = new BtreeNode<char>('C', new BtreeNode<char>('F', new BtreeNode<char>('L'), new BtreeNode<char>('M')),
                                                      new BtreeNode<char>('G', new BtreeNode<char>('N'), new BtreeNode<char>('O')));

	BtreeNode<char>* root = new BtreeNode<char>('A', nodeB, nodeC);
    print_tree(root);
	
	test_lowest_common_ancestor_output(root, 'I', 'H');
	test_lowest_common_ancestor_output(root, 'B', 'F');
	test_lowest_common_ancestor_output(root, 'A', 'O');
	test_lowest_common_ancestor_output(root, 'A', 'Z');

	cout << endl;

	delete root;
}

void
test_is_bst()
{
	{
		BtreeNode<int>* root = new BtreeNode<int>(4, new BtreeNode<int>(2, new BtreeNode<int>(1), new BtreeNode<int>(3)),
                                                     new BtreeNode<int>(6, new BtreeNode<int>(5), new BtreeNode<int>(7)));

		print_tree(root);
		cout << boolalpha << "Is bst? " << is_bst(root) << endl;

		delete root;
	}

	{
		BtreeNode<int>* root = new BtreeNode<int>(3, new BtreeNode<int>(2, new BtreeNode<int>(1), new BtreeNode<int>(4)), 
                                                     new BtreeNode<int>(6, new BtreeNode<int>(5), new BtreeNode<int>(7)));

		print_tree(root);
		cout << boolalpha << "Is bst? " << is_bst(root) << endl;

		delete root;
	}

	{
		BtreeNode<int>* root = new BtreeNode<int>(4, new BtreeNode<int>(2, new BtreeNode<int>(3), new BtreeNode<int>(1)), 
                                                     new BtreeNode<int>(6, new BtreeNode<int>(5), new BtreeNode<int>(7)));

		print_tree(root);
		cout << boolalpha << "Is bst? " << is_bst(root) << endl;

		delete root;
	}

	{
		BtreeNode<int>* root = new BtreeNode<int>(1, nullptr, new BtreeNode<int>(2, nullptr, new BtreeNode<int>(3, nullptr, new BtreeNode<int>(4))));

		print_tree(root);
		cout << boolalpha << "Is bst? " << is_bst(root) << endl;

		delete root;
	}
}

int main() {
    cout << "---Part1: Binary Tree---" << endl;
    vector<BtreeNode<int> *> trees;
    vector<string> testCaseNames;
    BtreeNode<int> *tree;

    // Tree 1
    tree = new BtreeNode<int>(2, new BtreeNode<int>(1));
    trees.push_back(tree);
    testCaseNames.push_back("Tree 1");

    // Tree 2
    tree = new BtreeNode<int>(2, new BtreeNode<int>(1), new BtreeNode<int>(3));
    trees.push_back(tree);
    testCaseNames.push_back("Tree 2");

    // Tree 3
    tree = new BtreeNode<int>(0, new BtreeNode<int>(1, NULL, new BtreeNode<int>(2)), new BtreeNode<int>(3));
    trees.push_back(tree);
    testCaseNames.push_back("Tree 3");

    for (int i = 0; i < trees.size(); i++) {
        cout << "# " << testCaseNames[i] << endl;
        print_tree(trees[i]);
        cout << endl;

        cout << "Nodes: " << count_nodes(trees[i]) << endl;
        cout << "Height: " << tree_height(trees[i]) << endl;


        BtreeNode<int>* mirror_tree = mirror(trees[i]);
        cout<< "Mirroring: ";
        print_tree(mirror_tree);

        cout<<endl;
        delete mirror_tree;
    }

    // Is complete
    cout<< "---Part2: isComplete---"<<endl;
    for(int i = 0; i < trees.size(); i++){
        cout << "# " << testCaseNames[i] << endl;
        cout << "Is Complete: " << (is_complete(trees[i]) ? "Yes" : "No") << endl;
    }

    // Clean up
    for(int i = 0; i < trees.size(); i++){
        delete trees[i];
    }
    cout<<endl;
    cout << "---Part3: Binary Search Tree---" << endl;
    BinarySearchTree<int, float>* bst = new BinarySearchTree<int, float>();
    // max value in an empty bst
    bst->print_max();

    bst->insert(4, 2.0);
    bst->insert(3, 4.0);
    bst->insert(3, 2.0);
    bst->insert(5, 3.5);
    bst->insert(6, 4.0);
    bst->insert(5, 4.0);

    bst->print_tree();
    cout<<endl;
    std::cout << std::boolalpha;
    cout<<"Contains (3,4.0): "<<bst->contains(3,4.0)<<endl;
    cout<<"Contains (5,3.5): "<<bst->contains(5,3.5)<<endl;
    cout<<"Contains (5,2.0): "<<bst->contains(5, 2.0)<<endl;

    cout<<endl;
    bst->print_max();
    delete bst;
    
    cout<<endl;
    cout << "---Part4: Lowest Common Ancestor---" << endl;
    test_lowest_common_ancestor();

    cout << "---Part5: Is BST---" << endl;
    test_is_bst();

    return 0;
}