//
// Binary Tree Node Class.
// Should not be modified.
//

#ifndef BTREE_NODE_H
#define BTREE_NODE_H

template<class T>
class BtreeNode {
public:
    BtreeNode(const T &x, BtreeNode *L = 0, BtreeNode *R = 0) :
            data(x), left(L), right(R) {}

    ~BtreeNode() {
        delete left;
        delete right;
    }

    const T &get_data() const { return data; }

    BtreeNode *get_left() const { return left; }

    BtreeNode *get_right() const { return right; }

    const T find_max() const {
        T result = data;

        if (left) {
            T left_result = left->find_max();
            if (left_result > result)
                result = left_result;
        }

        if (right) {
            T right_result = right->find_max();
            if (right_result > result)
                result = right_result;
        }

        return result;
    }

    const T find_min() const {
        T result = data;

        if (left) {
            T left_result = left->find_min();
            if (left_result < result)
                result = left_result;
        }

        if (right) {
            T right_result = right->find_min();
            if (right_result < result)
                result = right_result;
        }

        return result;
    }
    bool is_ancestor_of(const T& data) const {
        if (data == this->data)
            return true;

        if (left)
            if (left->is_ancestor_of(data))
                return true;

        if (right)
            if (right->is_ancestor_of(data))
                return true;

        return false;
    }

private:
    T data;
    BtreeNode *left;
    BtreeNode *right;
};

#endif //BTREE_NODE_H
