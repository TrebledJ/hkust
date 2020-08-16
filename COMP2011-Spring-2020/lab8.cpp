//
//  lab8.cpp
//  Copyright © 2020 TrebledJ. All rights reserved.
//

#include "lab_helpers.hpp"

#include <cctype>
#include <exception>
#include <fstream>
#include <iostream>
#include <thread>

#include <algorithm>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <vector>


using namespace std;


#define AUTOLOADSAVE    false       //  set false so that you can manually load/save stuff
const string DEFAULT_DB_FILENAME = "../../lab8_data.txt";   //  where the load/saves normally happen

/**
 Miscellaneous helper functions
 */
void delay_short(unsigned n = 2) { this_thread::sleep_for(n * 100ms); }
void delay_long(unsigned n = 1) { this_thread::sleep_for(n * 1000ms); }
void confirm() { cout << " (enter to continue) "; cin.get(); }
bool is_space(unsigned char ch) { return isspace(ch); }
string trim(string str)
{
    str.erase(str.begin(), find_if_not(str.begin(), str.end(), is_space));
    str.erase(find_if_not(str.rbegin(), str.rend(), is_space).base(), str.end());
    return str;
}


/**
 A robust database abstracted to handle local data management + file io.
 
 class Database
    type Student = string
    type Course = string
    type Record = pair<Student, Course>
 
    Database(string const& filename) -> Database
 
    verify_student(Student const&) -> void
    add(Student const&, Course const&) -> bool
    drop(Student const&, Course const&) -> void
    get_courses_for(Student const&) const -> set<Course>
    get_students() const -> vector< pair<Student, int> >
    clear_student(Student const&) -> void
    clear_database() -> void
    load(string const& filename = default) -> void
    commit(string const& filename = default) -> void
 */
class Database
{
public:
    using Student = string;
    using Course = string;
    using Record = pair<Student, Course>;
    
    struct database_error : public runtime_error { database_error(string const& str = "") : runtime_error{str} {} };
    struct student_not_found : database_error { student_not_found(Student const& student) : database_error{"student [" + student + "] not found"} {} };
    struct course_duplicate : database_error { course_duplicate(Student const& student, Course const& course) : database_error{"course [" + course + "] already added for student [" + student + "]"} {} };
    struct course_not_found : database_error { course_not_found(Student const& student, Course const& course) : database_error{"course [" + course + "] not found for student [" + student + "]"} {} };
    struct file_open_error : database_error { file_open_error(string const& filename) : database_error("an error occurred when opening file '" + filename + "'") {} };

private:
    map<Student, set<Course>> data;

public:
    void verify_student(Student const& student) const
    {
        if (data.find(student) == data.end())
            throw student_not_found{student};
    }
    
    /**
     @return Whether the course is the first course added for a student
     */
    bool add(Student const& student, Course const& course)
    {
        auto [it, inserted] = data[student].insert(course);
        if (!inserted)
            throw course_duplicate{student, course};
        
        if (AUTOLOADSAVE) commit();
        return data[student].size() == 1;
    }
    
    void drop(Student const& student, Course const& course)
    {
        if (data.find(student) == data.end())
            throw student_not_found{student};
        
        auto it = data[student].find(course);
        if (it == data[student].end())
            throw course_not_found{student, course};
        
        data[student].erase(it);
        if (AUTOLOADSAVE) commit();
    }
    set<Course> get_courses_for(Student const& student) const
    {
        if (data.find(student) == data.end())
            throw student_not_found{student};
        return data.at(student);
    }
    vector<pair<Student, int>> get_students() const
    {
        vector<pair<Student, int>> students; students.reserve(data.size());
        for (auto const& [st, crs] : data)
            students.push_back({st, crs.size()});
        return students;
    }
    void clear_student(Student const& student)
    {
        auto it = data.find(student);
        if (it == data.end())
            throw student_not_found{student};
        data[student].clear();
        if (AUTOLOADSAVE) commit();
    }
    void clear_database()
    {
        data.clear();
        if (AUTOLOADSAVE) commit();
    }
    
    void load(string const& filename = DEFAULT_DB_FILENAME)
    {
        ifstream in{filename};
        if (!in) throw file_open_error{filename};
            
        string student;
        string buffer;
        while (getline(in, buffer))
        {
            if (buffer.empty())
                continue;
            
            if (isspace(buffer.front()))
                data[student].insert(trim(buffer));
            else
            {
                student = buffer;
                data[student] = set<Course>{};
            }
        }
    }
    void commit(string const& filename = DEFAULT_DB_FILENAME) const
    {
        ofstream out{filename};
        if (!out) throw file_open_error{filename};
        
        Printer file{out, ""};
        
        for (auto it = data.begin(); it != data.end(); ++it)
        {
            string student = it->first;
            file.print(student);
            for (auto const& crs : it->second)
                file.print('\t', crs);
        }
    }
};



struct Point {
    int x, y;
    friend std::istream& operator>> (std::istream& is, Point& p) { return is >> p.x >> p.y; }
    friend std::ostream& operator<< (std::ostream& os, Point const& p) { return os << p.x << " " << p.y; }
};

/**
 * The meat of the program
 */
class Application
{
    static const string ADD_DROP_MENU;
    static const bool VERIFY = true;
    
    struct empty_string_error : runtime_error { empty_string_error(string const& subject) : runtime_error{subject + " cannot be empty"} {} };
    
    Scanner sc;
    Printer print;
    Database db;
    
    optional<Database::Student> selected_student;
    optional<Database::Course> selected_course;
    
public:
    Application()
        : print{std::cout, ""}
    {
    }
    
    bool load_database()
    {
        try { db.load(); }
        catch (Database::database_error& e)
        {
            print(e.what());
            return false;
        }
        return true;
    }
    
    int run()
    {
        if (AUTOLOADSAVE && !load_database())
            return 0;
        
        while (1)
        {
            print(ADD_DROP_MENU);
            
            char opt = sc.get_a<char>("(option) >>> ");
        
            try
            {
                switch (opt)
                {
                case '1': case 'A': case 'a':   //  add
                {
                    auto [student, course] = get_record();
                    auto is_first = db.add(student, course);
                    print(); print("Success: Added course [", course, "] for student [", student, "]");
                    if (is_first)
                        print("Note: This is the first course of this student");
                    delay_long(2);  //  this delay stuff is here to give the user more time
                                    //  to read the result and messages
                    break;
                }
                case '2': case 'D': case 'd':   //  drop
                {
                    auto [student, course] = get_record(VERIFY);
                    db.drop(student, course);
                    print(); print("Success: Dropped course [", course, "] for student [", student, "]");
                    delay_long(2);
                    break;
                }
                case '3': case ',':     //  list (students)
                {
                    auto students = db.get_students();
                    print(); print("Students (", students.size(), "):");
                    if (students.empty())
                        print("    (no students)");
                    else
                    {
                        for (auto const& [student, nb_courses] : students)
                            print("    ", student, " (", nb_courses, ")");
                    }
                    
                    print(); delay_long();
                    confirm();
                    break;
                }
                case '4': case '.':     //  list (courses)
                {
                    auto student = get_student(VERIFY);
                    auto courses = db.get_courses_for(student);
                    print(); print("Course List for [", student, "] (", courses.size(), "):");
                    if (courses.empty())
                        print("    (no courses)");
                    else
                    {
                        for (auto const& course : courses)
                            print("    ", course);
                    }
                    print(); delay_long();
                    confirm();
                    break;
                }
#ifdef AUTOLOADSAVE
                case '5': case 'O': case 'o':   //  load
                {
                    auto filename = sc.get_line("(filename) >>> ");
                    print(); print("Loading data from ", filename, "...");
                    db.load(filename);
                    print("Success!");
                    delay_long(2);
                    break;
                }
                case '6': case 'P': case 'p':   //  save
                {
                    auto filename = sc.get_line("(filename) >>> ");
                    print(); print("Saving data to ", filename, "...");
                    db.commit(filename);
                    print("Success!");
                    delay_long(2);
                    break;
                }
#endif
                case '7': case '[':     //  clear (student)
                {
                    auto student = get_student(VERIFY);
                    db.clear_student(student);
                    print(); print("Success: Cleared all courses for student [", student, "]");
                    delay_long(2);
                    break;
                }
                case '8': case ']':     //  clear (database)
                {
                    db.clear_database();
                    print(); print("Success: Cleared all courses for all students");
                    delay_long(2);
                    break;
                }
                case '0': case 'Q': case 'q':   //  quit
                {
                    print(); print("Good-bye! o/");
                    return 0;
                }
                default:
                    print("'", opt, "' is not a recognised option");
                    break;
                }
            }
            catch (runtime_error& e)
            {
                print(); print("Error: ", e.what());
                delay_long(2);
            }
            
            print();
        }
    }
    
private:
    Database::Student get_student(bool verify = false)
    {
        Database::Student student = trim(sc.get_line("(student) >>> "));
        if (student.empty())
            throw empty_string_error{"student"};
        if (verify)
            db.verify_student(student);
        return student;
    }
    
    Database::Record get_record(bool verify_student = false)
    {
        Database::Student student = get_student(verify_student);
        Database::Course course = trim(sc.get_line("(course) >>> "));
        if (course.empty())
            throw empty_string_error{"course"};
        return make_tuple(student, course);
    }
};


const string Application::ADD_DROP_MENU = ""
    "Add-Drop Course Menu\n"
    "   1/A - Add\n"
    "   2/D - Drop\n"
    "   3/, - List (Students)\n"
    "   4/. - List (Courses)\n"
#ifdef AUTOLOADSAVE
    "   5/O - Load\n"
    "   6/P - Save\n"
#endif
    "   7/[ - Clear (Student)\n"
    "   8/] - Clear (Database)\n"
    "   0/Q - Quit"
;

int main()
{
    Application app;
    return app.run();
}
