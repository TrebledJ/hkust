//submit this file
//you do NOT need to include any header in this file
//just write your HashTable implementation here right away


// Couldn't modify the class definition, so went with a macro instead -.-
// Otherwise I would've written a ton of helper functions.
#define next(k, i)  (mode == Mode::LINEAR ? h(k) + i : mode == Mode::QUADRATIC ? h(k) + i*i : h(k) + i*h2(i))


template <typename K, typename T>
HashTable<K,T>::HashTable(int m, int (*h)(K), int (*h2)(int), Mode mode, double loadLimit, bool isReferenceOnly)
    : m{m}, mode{mode}, table{new Cell[m]}, h{h}, h2{h2}, count{0}, loadLimit{loadLimit}, isReferenceOnly{isReferenceOnly}
{
    for (int i = 0; i < m; i++)
    {
        table[i].data = nullptr;
        table[i].status = CellStatus::EMPTY;
    }
}

template <typename K, typename T>
HashTable<K,T>::~HashTable()
{
    for (int i = 0; i < m; i++)
    {
        if (table[i].status == CellStatus::ACTIVE && !isReferenceOnly)
            delete table[i].data;
    }
    delete[] table;
}

template <typename K, typename T>
HashTable<K,T>::HashTable(const HashTable& another)
    : table{nullptr}
{
    *this = another;
}

template <typename K, typename T>
void HashTable<K,T>::operator=(const HashTable& another)
{
    if (table)
    {
        for (int i = 0; i < m; i++)
        {
            if (table[i].status == CellStatus::ACTIVE && !isReferenceOnly)
                delete table[i].data;
        }
        delete[] table;
    }

    m = another.m;
    mode = another.mode;

    table = new Cell[m];
    for (int i = 0; i < m; i++)
    {
        if (another.table[i].status == CellStatus::ACTIVE)
        {
            table[i].key = another.table[i].key;
            table[i].data = new T{*another.table[i].data};
        }
        else
        {
            table[i].data = nullptr;
        }
        
        table[i].status = another.table[i].status;
    }

    h = another.h;
    h2 = another.h2;
    count = another.count;
    loadLimit = another.loadLimit;
    isReferenceOnly = another.isReferenceOnly;
}

template <typename K, typename T>
int HashTable<K,T>::add(K key, T* data)
{
    if ((count + 1) > m * loadLimit)
    {
        // Rehash.
        Cell* new_table = new Cell[m*2];
        for (int i = 0; i < m*2; i++)
            new_table[i].status = CellStatus::EMPTY;
        
        // Insert data into new table.
        for (int i = 0; i < m; i++)
        {
            if (table[i].status == CellStatus::ACTIVE)
            {
                // Find new index.
                int hk = next(table[i].key, 0) % (m*2);
                for (int j = 1; new_table[hk].status == CellStatus::ACTIVE; j++)
                    hk = next(table[i].key, j) % (m*2);
                
                // Move new data.
                new_table[hk].key = table[i].key;
                new_table[hk].data = table[i].data;
                new_table[hk].status = CellStatus::ACTIVE;
            }
        }

        // Cleanup.
        delete[] table;
        std::swap(table, new_table);
        m *= 2;
    }

    // Find a deleted/empty (inactive) spot to insert.
    int i = 0;
    int hk = next(key, 0) % m;
    while (i++ < m && table[hk].status == CellStatus::ACTIVE)
        hk = next(key, i) % m;

    if (i < m)
    {
        // Set key/data and make the cell active.
        table[hk].key = key;
        table[hk].data = data;
        table[hk].status = CellStatus::ACTIVE;
        count++;
    }

    // Return #collisions or failure.
    return i < m ? i-1 : -1;
}


template <typename K, typename T>
bool HashTable<K,T>::remove(K key)
{
    // Iterate through collisions until an active cell is found with the same key.
    for (int i = 0, hk = h(key) % m; i < m; i++, hk = next(key, i) % m)
        if (table[hk].status == CellStatus::ACTIVE && table[hk].key == key) // Important that this order of checks is kept.
        {
            if (!isReferenceOnly)
                delete table[hk].data;
            table[hk].data = nullptr;
            table[hk].status = CellStatus::DELETED;
            count--;
            return true;
        }
    return false;
}


template <typename K, typename T>
T* HashTable<K,T>::get(K key) const
{
    for (int i = 0, hk = h(key) % m; i < m; i++, hk = next(key, i) % m)
        if (table[hk].status == CellStatus::ACTIVE && table[hk].key == key)
            return table[hk].data;
    return nullptr; // 404 not found.
}
