//
//  lab_helpers.hpp
//  Copyright © 2020 TrebledJ. All rights reserved.
//

//
//  Includes functions for reading input and writing output.
//  These are just robust readers/printers.
//  Use with C++17.
//
//  Usage:
//      Scanner scanner;
//      auto [width, height] = scanner.get_input<unsigned, unsigned>("Enter width and height: ");
//
//      Printer print;
//      print("width =", width, "|| height =", height);
//

#ifndef LAB_HELPERS_H
#define LAB_HELPERS_H

#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <tuple>


/**
 * @brief   A class for reading input from a stream. Kinda modelled after Python's input() and Java's Scanner.
 *           * Outputs a prompt
 *           * Checks if input has failed (if failed, will prompt again)
 *           * Turn interaction (i.e. prompts) off by setting interactive to false
 *              (e.g. scanner.set_interactive(false); )
 *           * Sometimes you might accidentally input more information than necessary.
 *              (e.g. program only wants 2 ints but you give it 3)
 *             Set `ignore_after` to TRUE to always ignore additional input given to the same instruction.
 *             Of course, the above only applies if you're inputting stuff interactively.
 *             If you're copy-pasting a set of input, then set `ignore_after` to FALSE.
 *
 *          Functions for reading input:
 *              get_a<Type>(string prompt, bool ignore_after)               -> Type
 *                  Reads one input of the given type T
 *
 *              get_line(string prompt, char delimiter, bool ignore_after)  -> string
 *                  Reads one line of input. Stops at `delimiter`
 *
 *              get<Types...>(string prompt, bool ignore_after)             -> tuple<Types...>
 *                  Reads a series of inputs of the given types Ts
 *
 * @example     Scanner sc;
 *              auto [x, y] = sc.get<unsigned, unsigned>("(<x> <y>) >>> ");
 *              auto cmd = sc.get_a<char>("(<cmd>) >>> ");
 *              auto line = sc.get_line("(<text>) >>> ");
 *
 *              struct Point {
 *                  int x, y;
 *                  friend std::istream& operator>> (std::istream& is, Point& p) { return is >> p.x >> p.y; }
 *              };
 *              auto point = sc.get_a<Point>("(<x> <y>) >>> "):
 *              auto [x, y] = sc.get_a<Point>("(<x> <y>) >>> "):
 */
class Scanner
{
public:
    Scanner(std::istream& is = std::cin, bool interactive = true);
    
    /// modifiers:
    Scanner& set_interactive(bool interactive);
    
    void ignore_all();
    
    /**
     * @brief   Reads one input of the given type T
     */
    template<class T>
    T get_a(std::string const& prompt = "",
            std::optional<bool> ignore_after = std::nullopt);
    
    /**
     * @brief   Reads one line of input. Stops at `delimiter`
     */
    std::string get_line(std::string const& prompt = "",
                         char delimiter = '\n',
                         std::optional<bool> ignore_after = std::nullopt);

    /**
     * @brief   Reads a series of inputs of the given types Ts
     */
    template<class ...Ts>
    std::tuple<Ts...> get(std::string const& prompt = "",
                          std::optional<bool> ignore_after = std::nullopt);

private:
    bool m_interactive;
    std::istream& m_stream;
    
private:
    void bad_read();
    
    void do_prompt(std::string const& prompt);
    
    template<class T>
    bool saferead(T& t);
    
    template<class ...Ts, size_t ...Ints>
    std::tuple<Ts...> get_impl(std::string const& prompt,
                               std::optional<bool> ignore_after,
                               std::index_sequence<Ints...> seq);
    
    template<class ...Ts>
    friend std::istream& operator>> (std::istream& is, std::tuple<Ts...>& tup);
};


/**
 * @brief   A class for printing output. Kinda modelled after Python's print.
 *
 * @example     Printer print;
 *              print("Hello world");       //  console: Hello world
 *              print(1, '+', 1, '=', 1+1); //  console: 1 + 1 = 2
 *
 *              struct Point {
 *                  int x, y;
 *                  friend std::ostream& operator<< (std::ostream& os, Point const& p) { return os << p.x << " " << p.y; }
 *              };
 *
 *              Point p{10, 40};
 *              print("Point:", p);    //  console: Point: 10 40
 */
class Printer
{
public:
    Printer(std::ostream& os = std::cout,
            std::string const& sep = " ",
            std::string const& end = "\n");
    
    Printer& sep(std::string const& sep);
    Printer& end(std::string const& end);
    void start();
    void stop();
    
    /// print functions:
    void print();
    
    template<class T>
    void print(T const& arg);
    
    template<class T, class ...Ts>
    void print(T const& arg, Ts const&... args);
    
    /// convenience function for `cout.width(size); cout << ch;`
    void printw(int size, char ch = ' ');
    void printw(int size, std::string const& s = " ");
    
    /// convenience functions/overloads:
    void operator() (); //  same as print()
    
    template<class ...Ts>
    void operator() (Ts const&... args);    //  same as print(...)
    
    void w(int size, std::string const& s = " ");    //  same as printw(...)
    void w(int size, char ch = ' ');
    
private:
    std::ostream& m_stream;
    std::string m_sep;
    std::string m_end;
    bool m_active;
};


/***********/
/* Scanner */
/***********/

inline
Scanner::Scanner(std::istream& is, bool interactive)
    : m_stream{is}
    , m_interactive{interactive}
{
}

/// modifiers:
inline Scanner& Scanner::set_interactive(bool interactive) { this->m_interactive = interactive; return *this; }

inline void Scanner::ignore_all() { m_stream.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); }

template<class T> inline
T Scanner::get_a(std::string const& prompt,
                 std::optional<bool> ignore_after)
{
    return std::get<0>(get_impl<T>(prompt, ignore_after, std::index_sequence_for<T>()));
}

inline
std::string Scanner::get_line(std::string const& prompt,
                              char delimiter,
                              std::optional<bool> ignore_after)
{
    std::string result;
    do_prompt(prompt);
    while (!std::getline(m_stream, result, delimiter))
    {
        bad_read();
        do_prompt(prompt);
    }
    return result;
}

template<class ...Ts> inline
std::tuple<Ts...> Scanner::get(std::string const& prompt,
                               std::optional<bool> ignore_after)
{
    return get_impl<Ts...>(prompt, ignore_after, std::index_sequence_for<Ts...>());
}

inline
void Scanner::bad_read()
{
    m_stream.clear();
    ignore_all();
    if (m_interactive)
        std::cout << "input rejected" << std::endl << std::endl;
}

inline
void Scanner::do_prompt(std::string const& prompt) { if (m_interactive) std::cout << prompt; }

template<class T> inline
bool Scanner::saferead(T& t)
{
    if (!(m_stream >> t))
    {
        bad_read();
        return false;
    }
    return true;
}

template<class ...Ts, size_t ...Ints> inline
std::tuple<Ts...> Scanner::get_impl(std::string const& prompt,
                                    std::optional<bool> ignore_after,
                                    std::index_sequence<Ints...> seq)
{
    std::tuple<Ts...> tup;
    //  read input, exit when reads are all OK (i.e. true)
    do { do_prompt(prompt); } while (!(saferead(std::get<Ints>(tup)) && ...));
    if (ignore_after.value_or(m_interactive)) ignore_all();
    
    return tup;
}

template<class ...Ts>
std::istream& operator>> (std::istream& is, std::tuple<Ts...>& tup)
{
    tup = Scanner{is}.get<Ts...>();
}


/***********/
/* Printer */
/***********/

inline
Printer::Printer(std::ostream& os,
        std::string const& sep,
        std::string const& end)
    : m_stream{os}, m_sep{sep}, m_end{end}, m_active{true}
{
}

inline Printer& Printer::sep(std::string const& sep) { this->m_sep = sep; return *this; }
inline Printer& Printer::end(std::string const& end) { this->m_end = end; return *this; }
inline void Printer::start() { m_active = true; }
inline void Printer::stop() { m_active = false; }

inline void Printer::print() { if (m_active) m_stream << m_end; }

template<class T> inline
void Printer::print(T const& arg) { if (m_active) m_stream << arg << m_end; }

template<class T, class ...Ts> inline
void Printer::print(T const& arg, Ts const&... args) { if (m_active) { m_stream << arg << m_sep; print(args...); } }

inline void Printer::printw(int size, char ch) { printw(size, std::string(1, ch)); }
inline void Printer::printw(int size, std::string const& s) { if (m_active) { std::cout.width(size); std::cout << s; } }

inline
void Printer::operator() () { print(); }

template<class ...Ts> inline
void Printer::operator() (Ts const&... args) { print(args...); }

inline void Printer::w(int size, char ch) { printw(size, ch); }
inline void Printer::w(int size, std::string const& s) { printw(size, s); }

#endif
