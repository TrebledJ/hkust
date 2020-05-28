//do NOT modify this file
//do NOT submit this file
#include <fstream>
#include <cstring>
#include <iostream>
#include <cstdlib>
#include <iomanip>


enum Stat {CASESRAW, DEATHSRAW, CASESPOP, DEATHSPOP, CASESAREA, DEATHSAREA, MORTALITY};

namespace Util
{
    std::string to_string(Stat stat);
    std::string filename(Stat stat);
}

class DayStat
{
private:
    double cases, deaths;
public:
    DayStat();
    DayStat(int _cases, int _deaths);
    DayStat(const DayStat &d, double denominator);
    double mortalityRate() const;
    double getcases() const;
    double getdeaths() const;
};

class Region
{
private:
    DayStat *raw;   //  dynamic array storing raw input data
    char *name;
    int population;
    int area;
    int nday;

    DayStat *normPop;
    DayStat *normArea;
    double *mortality;

public:
    Region();
    Region(Region const&) = delete;
    Region(Region&&) = delete;
    ~Region();
    Region& operator= (Region const&) = delete;
    Region& operator= (Region&&) = delete;
    void readline(char *line);
    void normalizeByPopulation();
    void normalizeByArea();
    void computeMortalityRate();
    void write(Stat stat) const;
};

int readcsv(Region*& region, const char* csvFileName);
void writecsvs(const Region* region, int nRegions);
