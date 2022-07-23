/*
 * COMP2011 (Spring 2020) Assignment 3
 *
 * Student name: [redacted]
 * Student ID: [redacted]
 * Student email: [redacted]
 */


//submit this file ONLY
//if you need to write your own helper functions, write the functions in this file
//again, do NOT include additional libraries and make sure this todo.cpp can be compiled with the unmodified versions of the other given files in our official compilation environment



#include "given.h"
#include "todo.h"


//
//  Helpers
//


namespace StringHelper
{
    /// ! allocates [] memory
    char* copy(char* str, size_t length)
    {
        if (length == string::npos)
            length = strlen(str);
        char* ret = new char[length + 1];
        strcpy(ret, str);
        return ret;
    }
    
    /// ! deallocates [] memory
    void debilitate(char* str)
    {
        if (str)
            delete[] str;
    }
    
    /// ! deallocates [] memory
    void debilitate2d(char** arr, int length)
    {
        if (!arr) return;
        
        for (int i = 0; i < length; ++i)
            debilitate(arr[i]);
        delete[] arr;
    }
    
    /// ! allocates [] memory
    char* append(const char* front, const char* end)
    {
        char* new_string = new char[strlen(front) + strlen(end) + 1];
        strcpy(new_string, front);
        strcat(new_string, end);
        return new_string;
    }
    
    bool equals(const char* a, const char* b)
    {
        return a && b && strcmp(a, b) == 0;
    }
}

class CSVLineChewieMuncher
{
    const char* m_line = nullptr;
    int m_length = 0;
    int m_index = 0;
    
public:
    /// constructors:
    CSVLineChewieMuncher(const char* line, int length = -1)
    {
        if (length == -1)
            length = int(strlen(line));
        
        m_line = line;
        m_length = length;
        m_index = 0;
    }
    
    CSVLineChewieMuncher(CSVLineChewieMuncher const&) = delete;
    
    /// assignment:
    CSVLineChewieMuncher& operator= (CSVLineChewieMuncher const&) = delete;
    
    
    /// modifiers:
    /**
     * @return  A new string with the next csv token, if it exists; nullptr otherwise or if token is empty
     */
    char* next()
    {
        if (done())
            return nullptr;
        
        int begin = m_index;
        int end = end_of_field();
        m_index = end + (m_line[end] == '\0' ? 0 : 1);
        if (begin == end)
            return nullptr;
        if (begin < end-1 && m_line[begin] == '\"' && m_line[end-1] == '\"')
            return make_substring(begin+1, end-1);
        else
            return make_substring(begin, end);
    }

    int next_int()
    {
        char* str = next();
        int n = atoi(str);
        StringHelper::debilitate(str);
        return n;
    }
    
    /**
     * @brief   Discards the next n tokens
     */
    void discard(int n = 1)
    {
        for (; n > 0; n--)
        {
            if (m_line[m_index] == '\0')
                return;
            
            int end = end_of_field();
            m_index = end + (m_line[end] == '\0' ? 0 : 1);
        }
    }
    
    /// observers:
    /**
     * @return  Number of (comma-separated) fields in the line
     */
    int fields() const
    {
        if (strlen(m_line) == 0)
            return 0;
        
        int f = 1;
        const char* res = m_line;
        while ((res = strchr(res, ',')))
        {
            f++;
            res++;
        }
        return f;
    }
    
    bool done() const
    {
        return m_line[m_index] == '\0';
    }

private:
    /**
     * @param   end past-the-end index
     */
    char* make_substring(int begin, int end) const
    {
        const int size = end - begin;
        char* substr = new char[size + 1];
        strncpy(substr, m_line + begin, size);
        substr[size] = '\0';
        return substr;
    }
    
    int end_of_field() const
    {
        bool inside_quote = false;
        int end = m_index;
        while (m_line[end] != '\0' && (inside_quote || m_line[end] != ','))
        {
            if (m_line[end] == '\"')
                inside_quote = !inside_quote;
            
            ++end;
        }
        return end;
    }
};

namespace NodeHelper
{
    /// ! allocates memory
    Node* make_list(CSVLineChewieMuncher& line)
    {
        Node* head = nullptr;
        Node* curr = nullptr;
        for (auto day = 1; !line.done(); ++day)
        {
            int number = line.next_int();
            if (number == 0)
                continue;
            
            if (!curr)
                curr = new Node;
            else
            {
                curr->next = new Node;
                curr = curr->next;
            }
            curr->day = day;
            curr->number = number;
            
            if (!head) head = curr;
        }
        return head;
    }
    
    /// @return Next node
    Node* delete_node(Node* node)
    {
        Node* temp = node->next;
        delete node;
        return temp;
    }
        
    void normalise_choppy_chop(Node*& head, int threshold)
    {
        int sunshine_bucket = 1;
        Node* prev = nullptr;
        Node* curr = head;
        while (curr != nullptr)
        {
            if (curr->number >= threshold)
            {
                curr->day = (sunshine_bucket++);
                prev = curr;
                curr = curr->next;
                continue;
            }
            
            if (curr == head)   head = curr = delete_node(curr);
            else                curr = delete_node(curr);
            
            if (prev)           prev->next = curr;
        }
    }
    
    Node* get_next(Node* node, int n = 1)
    {
        return n == 0 ? node : (node ? get_next(node->next, n-1) : nullptr);
    }
    
    void remove_zeroes(Node* node)
    {
        Node* prev = nullptr;
        while (node != nullptr)
        {
            if (node->number == 0)
                prev->next = node = delete_node(node);
            else
            {
                prev = node;
                node = node->next;
            }
        }
    }
    
    void nday_growy_pew_pew(Node* node, int n)
    {
        Node* curr = node;
        for (int i = 0; i < n && node != nullptr; ++i)
        {
            int past_numero = curr->number;
            while (curr != nullptr)
            {
                curr = get_next(curr, n);
                if (curr == nullptr)
                    break;
                curr->number -= past_numero;
                past_numero = curr->number + past_numero;
            }
            
            curr = node = node->next;
        }
    }
    
    void delete_place(Place& place)
    {
        StringHelper::debilitate(place.region);
        StringHelper::debilitate(place.province);
        deallocateLinkedList(place.headNode);
    }
}

namespace MergeHelper
{
    int count_unique_regions(Place* places, int place_count)
    {
        int count = place_count;
        for (int i = 0; i < place_count; ++i)
            for (int j = i+1; j < place_count; ++j)
                if (StringHelper::equals(places[i].region, places[j].region))
                {
                    count--;
                    break;
                }
        return count;
    }
    
    int find_region(Place* places, int places_count, const char* region)
    {
        for (int i = 0; i < places_count; ++i)
            if (StringHelper::equals(places[i].region, region))
                return i;
        return -1;
    }
    
    /**
     * UB if not enough space in places.
     */
    void add_region(Place*& places, int& places_count, char*& region)
    {
        swap(places[places_count].region, region);
        places_count++;
    }
    
    /**
     * Does not copy tail nodes.
     */
    Node* copy_node(const Node* const node)
    {
        Node* ret = new Node;
        ret->day = node->day;
        ret->number = node->number;
        return ret;
    }
    
    void insert_head_node(Node*& head, Node* node)
    {
        if (!node)  return;
        if (!head) { head = node; return; }
        node->next = head;
        head = node;
    }
    
    void insert_node_after(Node*& target, Node* node)
    {
        if (!node)  return;
        if (!target) { target = node; return; }
        node->next = target->next;
        target->next = node;
    }
    
    void insert_node(Node*& source, const Node* const node)
    {
        //  Cases:
        //  0. Node already exists (by day)
        //  1. Insert in empty list
        //  2. Insert before list
        //  3. Insert after list
        //  4. Insert within list
        
        if (!node)  return;
        
        //  case 1: empty list
        if (!source) { source = copy_node(node); return; }
        
        //  case 2: before list
        if (node->day < source->day) { insert_head_node(source, copy_node(node)); return; }
        
        Node* curr = source;
        while (curr != nullptr)
        {
            if (node->day == curr->day)
            {
                //  case 0
                curr->number += node->number;
                return;
            }
            else if (node->day > curr->day)
            {
                if (!curr->next                                         //  case 3: after list
                    || (curr->next && node->day < curr->next->day)      //  case 4: within list
                    )
                {
                    insert_node_after(curr, copy_node(node));
                    return;
                }
                curr = curr->next;
            }
        }
    }
    
    void merge_node_list(Node*& source, Node* node)
    {
        while (node != nullptr)
        {
            insert_node(source, node);
            node = node->next;
        }
    }
}

void swap(Place& p, Place& q)
{
    swap(p.region, q.region);
    swap(p.province, q.province);
    swap(p.headNode, q.headNode);
}

    
//
//  Assignment 3
//


//given the csv header line (i.e., first row), return the number of dates
int getDateCount(char* headerLine)
{
    CSVLineChewieMuncher line{headerLine};
    return line.fields() - 4;
}

//given the csv header line (i.e., first row), return an array of dates which are stored in the order of their appearances, as a dynamic array of dynamic char arrays
//remember all char arrays should be null-terminated
char** getDates(char* headerLine)
{
    int date_count = getDateCount(headerLine);
    char** dates = new char*[date_count];
    CSVLineChewieMuncher line{headerLine};
    line.discard(4);
    for (int i = 0; i < date_count; ++i)
        dates[i] = line.next();
    
    return dates;
}

//given the dates array and the date count, return the day
//first date in the array is considered as day 1, and so on
//return 0 if the given date is not in the dates array
//note: this is actually not used in other tasks; it is by itself a separated test
//hint: you may use strcmp (see online documentation of it)
int getDay(char** dates, int dateCount, const char* date)
{
    for (int i = 1; i < dateCount+1; ++i)
        if (StringHelper::equals(dates[i-1], date))
            return i;
    return 0;
}

//given the csv lines (header row included) and the number of those
//return the dynamic array of places
//each of the place should have a linked list of dated numbers (i.e. you need to fill in day and number to each node)
//skip all days with 0 numbers (to reduce the use of memory, since there can be a lot of zeros)
//for the "day" integer in a linked list node, the first date in the header: "1/22/20" is considered as day 1, and the second date "1/23/20" is considered as day 2, etc.
//read the sample output and see what it means
//hint: the library function atoi may be useful for converting c-string to an integer, you may look for its online documentation
Place* getPlaces(char** csvLines, int csvLineCount)
{
    if (csvLineCount <= 1 || !csvLines)
        return nullptr;
    
    Place* places = new Place[csvLineCount-1];
    for (int i = 1; i < csvLineCount; ++i)
    {
        CSVLineChewieMuncher line{csvLines[i]};
        places[i-1].province = line.next();
        places[i-1].region = line.next();
        line.discard(2);
        places[i-1].headNode = NodeHelper::make_list(line);
    }
    
    return places;
}

//given the places array and the number of places,
//remove all provinces (deallocate all province character arrays, and set all province pointers to nullptr)
//merge all places (of any province) in the same region, to one single place
//by merging, it means, all numbers are added up for the same day
//therefore, the resulting new places array (will be given back via the places reference variable) is essentially a list of whole regions
//see sample output for examples
//the function returns the number of places in the new array
//note that the old array (likely bigger than the new array) shall be deallocated
//the last parameter "home" denotes the province that you have special interest in
//please see the webpage description for this part
int mergeAllProvinces(Place*& places, int placeCount, const char* home)
{
    int nb_regions = MergeHelper::count_unique_regions(places, placeCount);
    if (home && !StringHelper::equals(home, ""))
        nb_regions++;
    
    Place* merged = new Place[nb_regions];
    int merged_internal_size = 0;
    
    for (int i = 0; i < placeCount; ++i)
    {
        if (StringHelper::equals(places[i].province, home))
        {
            //  do special case (home)
            char* home_region = StringHelper::append(places[i].province, "(Home)");
            MergeHelper::add_region(merged, merged_internal_size, home_region);
            swap(merged[merged_internal_size-1].headNode, places[i].headNode);
        }
        else
        {
            int index = MergeHelper::find_region(merged, merged_internal_size, places[i].region);
            if (index == -1)
            {
                //  add region
                MergeHelper::add_region(merged, merged_internal_size, places[i].region);
                index = merged_internal_size - 1;
                
                //  move head node
                swap(merged[index].headNode, places[i].headNode);
            }
            else
            {
                //  update region
                MergeHelper::merge_node_list(merged[index].headNode, places[i].headNode);
            }
        }
    }
    
    //  cleanup and replace
    deallocatePlaces(places, placeCount);
    places = merged;
    return merged_internal_size;
}

//given the places array and the number of places
//give back a new places array and count (via the places reference variable and placeCount reference variable)
//the new places array is normalized in this sense:
//we consider the day with confirmed-cases number that is equal to or larger than "threshold", as day 1
//remove all nodes before day 1
//as a result, the day 1 node shall be the new head node
//the node after the day 1 node are day 2 node, day 3 node, and so on
//study samples for details
void normalizeDays(Place*& places, int& placeCount, int threshold)
{
    int idx_of_the_deceased_nations = -1;
    int number_of_nations_rejected_from_choppy_choppy_normalisation = 0;
    for (int i = 0; i < placeCount; ++i)
    {
        NodeHelper::normalise_choppy_chop(places[i].headNode, threshold);
        if (!places[i].headNode)
        {
            if (idx_of_the_deceased_nations == -1)
                idx_of_the_deceased_nations = i;
            
            number_of_nations_rejected_from_choppy_choppy_normalisation++;
        }
        else
        {
            if (idx_of_the_deceased_nations != -1)
            {
                //  swap with idx deceased
                swap(places[i], places[idx_of_the_deceased_nations]);
                idx_of_the_deceased_nations++;
            }
        }
    }
    
    for (int i = placeCount-1; i >= placeCount - number_of_nations_rejected_from_choppy_choppy_normalisation; --i)
    {
        NodeHelper::delete_place(places[i]);
    }
    
    placeCount -= number_of_nations_rejected_from_choppy_choppy_normalisation;
}

//given the places, the count of places, and n
//change the numbers of each day of the places to n-day growth
//for example, if a place has these numbers in day 1 to day 6:
//16,18,20,22,22,24
//and n is 3
//that means the numbers will be changed to [number of today] - [number of 3-days ago]
//so the numbers shall become
//16,18,20,6,4,4
//note that for the first 3 days, the numbers remain unchanged because there is no "3-days ago" for them
//study samples for more examples
void changeToNDayGrowth(Place* places, int placeCount, int n)
{
    for (int i = 0; i < placeCount; ++i)
    {
        NodeHelper::nday_growy_pew_pew(places[i].headNode, n);
        NodeHelper::remove_zeroes(places[i].headNode);
    }
}

//write the given places to a csv file, just like the sample output
void writeCSV(const char* csvFileName, Place* places, int placeCount)
{
    ofstream out{csvFileName};
    if (!out)
    {
        cout << "error opening file: " << csvFileName << endl;
        return;
    }
    
    for (int i = 0; i < placeCount; ++i)
    {
        out << places[i].region;
        Node* curr = places[i].headNode;
        while (curr != nullptr)
        {
            out << "," << curr->number;
            curr = curr->next;
        }
        out << endl;
    }
}

//deallocate all the csv lines given the array and count of lines
//hint: the given array is a dynamic array of dynamic arrays, deallocate them all
//hint: remember that you must use [] to deallocate a dynamic array
void deallocateCSVLines(char** csvLines, int csvLineCount)
{
    StringHelper::debilitate2d(csvLines, csvLineCount);
}

//deallocate the dates array given the array and the count of dates
void deallocateDates(char** dates, int dateCount)
{
    StringHelper::debilitate2d(dates, dateCount);
}

//deallocate all linked list nodes given the head node
void deallocateLinkedList(Node* node)
{
    while (node != nullptr)
        node = NodeHelper::delete_node(node);
}

//deallocate the whole places array given the pointer to it and the count of places
//note that you should also deallocate all dynamic char arrays (i.e. the country and province) and the linked list in each place
void deallocatePlaces(Place* places, int placeCount)
{
    for (int i = 0; i < placeCount; ++i)
        NodeHelper::delete_place(places[i]);
    
    delete[] places;
}

