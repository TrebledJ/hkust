//submit this file
//you do NOT need to include any header in this file
//just write your BST implementation here right away

using namespace std;


template <typename T>
BST<T>::BST(const BST& another)
{
    if (!another.root)
        return;
    
    root = new BSTNode<T>(another.root->key, another.root->value);
    root->left = another.root->left;
    root->right = another.root->right;
}

template <typename T>
bool BST<T>::isEmpty() const
{
    return !root;
}

template <typename T>
bool BST<T>::add(string key, T value)
{
    if (isEmpty())
    {
        root = new BSTNode<T>(key, value);
        return true;
    }
    else if (key < root->key)
        return root->left.add(key, value);
    else if (key > root->key)
        return root->right.add(key, value);
    else
        return false;
}

template <typename T>
bool BST<T>::remove(string key)
{
    if (isEmpty())
        return false;

    if (key < root->key)
        return root->left.remove(key);
    else if (key > root->key)
        return root->right.remove(key);
    else
    {
        if (root->left.root && root->right.root)
        {
            // Double children: replace with minimum node on right.
            const BST* min = root->right.findMin();
            // Copy data.
            root->key = min->root->key;
            root->value = min->root->value;
            // Remove key from subtree.
            root->right.remove(root->key);
        }
        else
        {
            BSTNode<T>* to_delete = root;  // Save node to delete.
            root = (root->left.isEmpty() ? root->right.root : root->left.root);

            // Nullify before deleting.
            to_delete->left.root = to_delete->right.root = nullptr;
            delete to_delete;
        }
        return true;
    }
}

template <typename T>
T* BST<T>::get(string key) const
{
    if (!root)
        return nullptr;
    else if (key < root->key)
        return root->left.get(key);
    else if (key > root->key)
        return root->right.get(key);
    else
        return &root->value;
}

template <typename T>
void BST<T>::getBetweenRangeHelper(const BST<T>* bst, string start, string end, list<T>* resultList) const
{
    if (!bst || !bst->root)
        return;

    const string& key = bst->root->key;
    getBetweenRangeHelper(&bst->root->left, start, end, resultList);
    if (start <= key && key <= end)
        resultList->push_back(bst->root->value);
    getBetweenRangeHelper(&bst->root->right, start, end, resultList);
}

template <typename T>
const BST<T>* BST<T>::findMin() const
{
    return !root ? nullptr : root->left.root ? root->left.findMin() : this;
}
