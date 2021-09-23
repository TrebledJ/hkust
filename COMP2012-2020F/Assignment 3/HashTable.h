#ifndef ASG3_HASHTABLE_H
#define ASG3_HASHTABLE_H



#include <iostream>

//collision resolving mode supported by our hash table
enum class Mode
{
   LINEAR, //linear probing
   QUADRATIC, //quadratic probing
   DOUBLE //double hashing
};

//HashTable is a customizable hash table class
//keys are of the K type
//values are of the T type
//open addressing and lazy deletion are always used (see lecture notes)
//however, the collision resolving is customizable
template <typename K, typename T>
class HashTable
{
private:
   enum class CellStatus
   {
       EMPTY, //empty cell
       ACTIVE, //active cell - that means the corresponding cell is storing an Cell
       DELETED //deleted cell - that means the cell is deleted and can be used; a deleted cell contains no usable content
   };
   struct Cell //a cell is a cell in the hash table
   {
       K key; //key
       T* data; //pointer to the data
       CellStatus status; //status of the cell
   };

   int m; //size of the table

   Mode mode; //collision resolving mode

   Cell* table; //the hash table of cells, a 1D dynamic array of cells

   int (*h)(K); //the function pointer that points to the hash function
   //note that this function shouldn't include the "mod m" part
   //as this function by itself has no knowledge of m
   //the "mod m" part will be applied when we use the hash function later in add/remove/get/etc.
   //for example, the hash function may be simply "k*k" for a given key k
    //however, when we use the hash function on a given key in add, we will further apply the "mod m" part
    //that is, we will calculate (k*k)%m to locate the cell to be used for insertion

    int (*h2)(int); //the function pointer that points to the second hash function (used in DOUBLE mode only)
    //again, this also doesn't include the "mod m" part

    int count; //count of active cells (existing and not deleted) in the table, should be 0 initially

    double loadLimit; //load limit, see the add function description

    bool isReferenceOnly; //if the isReferenceOnly is true, then you don't need to delete (i.e. deallocate) all the data inside all the Cells

public:
    //constructor
    //initialize all data members including the dynamic array
    HashTable(int m, int (*h)(K), int (*h2)(int), Mode mode, double loadLimit, bool isReferenceOnly = false);

    //deep copy constructor
    HashTable(const HashTable &another);

    //destructor
   //if isReferenceOnly == false
    //you must delete (i.e. deallocate) all the data inside all the Cells here and table
    //else you only need to delete (i.e. deallocate) the table
   ~HashTable();

   //deep assignment
   void operator=(const HashTable& another);

   //add the data with the given key
   //deep copy of the data is not needed
   //i.e., simple pointer assignment is OK when you add the data to the corresponding cell
   //if the number of occupied (i.e. active) cells would be larger than loadLimit*m after the new data entry,
   //perform rehashing first, before adding the data entry
   //please see the webpage for the description and example of rehashing
   //-----insertion part
   //if CellStatus is EMPTY or DELETED, then you can assign the key, data and status (ACTIVE) to the cell
   //else looking for next available index according to the collision resolving mode: Linear, Quadratic, Double
   //at the end of the function, you should return the number of collisions occurred (should be less than m)
   //however, if the number of collisions reach m while you attempt to insert the data
   //stop the attempt and simply return -1 to indicate that the data entry is not done
   int add(K key, T *data);

    //remove the entry, according to the given key, from the table
    //update the status of the corresponding cell and
    //if isReferenceOnly==false, then delete (i.e. deallocate) its data; else don't delete the data
    //return true after the deletion is done
    //simply return false if the key is not found
    bool remove(K key);

   //return the data given the key
   T* get(K key) const;

   //print the hash table to the ostream
   //given; just like the rest of this file, do not modify
   void print(std::ostream& os) const
   {
       for(int i=0; i<m; i++)
       {
           os << i << ": ";
           if(table[i].status == CellStatus::ACTIVE)
               os << "[Active]{" << table[i].key << "}";
           else if(table[i].status == CellStatus::EMPTY)
               os << "[Empty]";
           else if(table[i].status == CellStatus::DELETED)
               os << "[Deleted]";

           if(i!=m-1)
               os << " | ";
       }
       os << std::endl;
   }
};

#include "HashTableImplementation.h"



#endif //ASG3_HASHTABLE_H
