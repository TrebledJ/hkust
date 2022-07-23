//
//  COMP2011
//  lab10_modern.cpp
//  Copyright © 2020 TrebledJ. All rights reserved.
//

#include "lab_helpers.hpp"
#include <iostream>
#include <numeric>
#include <random>
#include <vector>


class LabTenBongBong    //  Bing bong bang boom
{
public:
    /// modifiers:
    void generate(int size)
    {
        //  create array
        v.clear();
        v.reserve(size);
        int leave_out = randint(size+1);
        for (int i = 0; i < size+1; ++i)
            if (i != leave_out)
                v.push_back(i);
        
        shuffle();
    }
    
    /// observers:
    int get_missing_number() const
    {
        return int(v.size()*(v.size()+1)/2 - std::accumulate(v.begin(), v.end(), 0ULL));
    }
    
    friend std::ostream& operator<< (std::ostream& os, LabTenBongBong const& bongbong)
    {
        for (int i = 0; i < bongbong.v.size(); i++)
            os << bongbong.v[i] << (i < bongbong.v.size()-1 ? " " : "");
        return os;
    }
    
private:
    static std::mt19937 gen;
    std::vector<int> v;
    
private:
    //  returns an int type in the range [0, n-1]
    static unsigned randint(unsigned n)
    {
        return n ? std::uniform_int_distribution<unsigned>(0, n-1)(gen) : 0;
    }
    
    void shuffle()
    {
        std::shuffle(v.begin(), v.end(), gen);
    }
};

std::mt19937 LabTenBongBong::gen {std::random_device{}()};


void solve(int size, int iter)
{
    LabTenBongBong bongbong;
    std::vector<int> memory;
    for (int i = 0; i < iter; ++i)
    {
        bongbong.generate(size);
        memory.push_back(bongbong.get_missing_number());
        
        std::cout << std::endl;
        std::cout << bongbong << std::endl;
        std::cout << "x" << i << ": " << memory.back() << std::endl;
    }
    
    std::cout << std::endl;
    for (int i = 0; i < memory.size(); ++i)
        std::cout << "x" << i << ": " << memory[i] << std::endl;
}

int get_int_in_range(int low, int high, std::string const& prompt)
{
    while (1)
    {
        Scanner sc;
        int n = sc.get_a<int>(prompt);
        if (low <= n && n <= high)
            return n;
        
        std::cout << "input rejected: out of range" << std::endl;
    }
    return 0;   //  won't be reached
}


constexpr int MAX_ITERATIONS = 5;
int main()
{
    int size = get_int_in_range(2, 100, "(length: [2, 100]) >>> ");
    int iter = get_int_in_range(1, MAX_ITERATIONS, "(iterations: [1, 5]) >>> ");
    solve(size, iter);
}
