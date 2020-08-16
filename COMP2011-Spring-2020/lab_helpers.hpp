//
//  lab_helpers.hpp
//  Copyright © 2020 TrebledJ. All rights reserved.
//

//  Includes functions for reading input and writing output
//  These are just robust readers/printers
//  Used most effectively with C++17
//
//  Usage:
//      Scanner scanner;
//      auto [width, height] = scanner.get_input<unsigned, unsigned>("Enter width and height: ");
//
//      Printer print;
//      print("width =", width, "|| height =", height);
//

#ifndef lab_helpers_h
#define lab_helpers_h

#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <tuple>


//  modelled after Python's input() and Java's Scanner
struct Scanner
{
    bool interactive;
    std::istream& stream;
    
    Scanner(std::istream& is = std::cin, bool interactive = true) : stream{is}, interactive{interactive}
    {}
    Scanner& set_interactive(bool interactive) { this->interactive = interactive; return *this; }
    
    void ignore_all() { stream.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); }
    
    template<class T>
    T get_a(std::string const& prompt = "",
            std::optional<bool> ignore_after = std::nullopt)
    {
        return std::get<0>(get_impl<T>(prompt, ignore_after, std::index_sequence_for<T>()));
    }
    
    std::string get_line(std::string const& prompt = "",
                         char delimiter = '\n',
                         std::optional<bool> ignore_after = std::nullopt)
    {
        std::string result;
        do_prompt(prompt);
        while (!std::getline(stream, result, delimiter))
        {
            bad_read();
            do_prompt(prompt);
        }
        return result;
    }

    template<class ...Ts>
    std::tuple<Ts...> get(std::string const& prompt = "",
                          std::optional<bool> ignore_after = std::nullopt)
    {
        return get_impl<Ts...>(prompt, ignore_after, std::index_sequence_for<Ts...>());
    }
    
private:
    void bad_read()
    {
        stream.clear();
        ignore_all();
        if (interactive)
            std::cout << "input rejected" << std::endl << std::endl;
    }
    
    void do_prompt(std::string const& prompt) { if (interactive) std::cout << prompt; }
    
    template<class T>
    bool saferead(T& t)
    {
        if (!(stream >> t))
        {
            bad_read();
            return false;
        }
        return true;
    }
    
    template<class ...Ts, size_t ...Ints>
    std::tuple<Ts...> get_impl(std::string const& prompt,
                               std::optional<bool> ignore_after,
                               std::index_sequence<Ints...> seq)
    {
        std::tuple<Ts...> tup;
        //  read input, exit when reads are all OK (i.e. true)
        do { do_prompt(prompt); } while (!(saferead(std::get<Ints>(tup)) && ...));
        if (ignore_after.value_or(interactive)) ignore_all();
        
        return tup;
    }
    
    template<class ...Ts>
    friend std::istream& operator>> (std::istream& is, std::tuple<Ts...>& tup)
    {
        tup = Scanner{is}.get<Ts...>();
    }
};


//  modelled after Python's print()
struct Printer
{
    std::ostream& stream;
    std::string sep;
    std::string end;
    bool active;
    
    Printer(std::ostream& os = std::cout,
            std::string const& sep = " ",
            std::string const& end = "\n")
    : stream{os}, sep{sep}, end{end}, active{true}
    {}
    
    Printer& set_sep(std::string const& sep = " ") { this->sep = sep; return *this; }
    Printer& set_end(std::string const& end = "\n") { this->end = end; return *this; }
    void start() { active = true; }
    void stop() { active = false; }
    
    void print() { if (active) stream << end; }
    
    template<class T>
    void print(T const& arg) { if (active) stream << arg << end; }
    
    template<class T, class ...Ts>
    void print(T const& arg, Ts const&... args) { if (active) { stream << arg << sep; print(args...); } }
    
    void printw(int size, char ch = ' ') { if (active) { std::cout.width(size); std::cout << ch; } }
    
    
    void operator() () { print(); }
    
    template<class ...Ts>
    void operator() (Ts const&... args) { print(args...); }
};


#endif /* lab_helpers_h */
