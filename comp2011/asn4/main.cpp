//do NOT modify this file
//do NOT submit this file

#include "todo.h"
using namespace std;

#define FOREACH_STAT(x)    for (Stat x : {CASESRAW, DEATHSRAW, CASESPOP, DEATHSPOP, CASESAREA, DEATHSAREA, MORTALITY})

//  we'll call this function after program output.
//  This will be our "handy-little-automated diff-checker"
bool check_output(std::string const& file1, std::string const& file2)
{
    std::ifstream ifs1{file1}, ifs2{file2};
    std::string a, b;
    while (1)
    {
        bool read1 = bool(std::getline(ifs1, a));
        bool read2 = bool(std::getline(ifs2, b));
        if (read1 != read2)
        {
            std::cout << "line number mismatch" << std::endl;
            return false;
        }
        if (a != b)
        {
            std::cout << "line content mismatch" << std::endl;
            return false;
        }
        if (!read1 || !read2)
            return true;
    }
    return true;
}


int main()
{
    std::cout << "Hi COVID-19 Outbreak Data Analyst 2.0!" << std::endl << std::endl;
    std::cout << "=======================================================" << std::endl;
    
    Region* regions;
    int nRegions = 0;
    std::cout << "Reading the CSV file..." << std::endl;
    const char* csvFileName = "assignment4/pa4_data.csv";
    const char* checkOutputPrefix = "assignment4/sample/";

    nRegions = readcsv(regions, csvFileName);
    
    std::cout << "=======================================================" << std::endl;
    std::cout << "Processing data.." << std::endl;

    for(int i = 0; i < nRegions; ++i){
        // normalize by population:
        regions[i].normalizeByPopulation();
        // normalize by area:
        regions[i].normalizeByArea();
        // compute mortality rate:
        regions[i].computeMortalityRate();
    }
    
    std::cout << "=======================================================" << std::endl;
    std::cout << "Writing to csvs..." << std::endl;
    writecsvs(regions, nRegions);

    std::cout << "=======================================================" << std::endl;
    std::cout << "Performing deallocations..." << std::endl;
    delete[] regions;

    std::cout << "=======================================================" << std::endl;
    std::cout << "Checking output..." << std::endl;
    bool check_flag = true;
    FOREACH_STAT(stat)
    {
        if (!check_output(Util::filename(stat), checkOutputPrefix + Util::to_string(stat) + ".csv"))
        {
            std::cout << "an inconsistency was found in " << Util::filename(stat) << std::endl;
            check_flag = false;
        }
    }
    if (check_flag)
        std::cout << "all good :)" << std::endl;
    
    std::cout << "=======================================================" << std::endl;
    std::cout << "Done! Please take care and stay healthy! :)" << std::endl;
    return 0;
}
