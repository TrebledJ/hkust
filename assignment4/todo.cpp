/*
 * COMP2011 (Spring 2020) Assignment 4
 *
 * Student name: [redacted]
 * Student ID: [redacted]
 * Student email: [redacted]
 */
//submit this file ONLY
//if you need to write your own helper functions, write the functions in this file
//again, do NOT include additional libraries and make sure this todo.cpp can be compiled with the unmodified versions of the other given files in our official compilation environment

#include "todo.h"


#define FOREACH_STAT(x)    for (Stat x : {CASESRAW, DEATHSRAW, CASESPOP, DEATHSPOP, CASESAREA, DEATHSAREA, MORTALITY})

const std::string FILENAME_PREFIX = "assignment4/output/";
const std::string FILENAME_SUFFIX = ".csv";


namespace Util
{
    std::string to_string(Stat stat)
    {
        switch (stat)
        {
        case CASESRAW:      return "CASESRAW";
        case DEATHSRAW:     return "DEATHSRAW";
        case CASESPOP:      return "CASESPOP";
        case DEATHSPOP:     return "DEATHSPOP";
        case CASESAREA:     return "CASESAREA";
        case DEATHSAREA:    return "DEATHSAREA";
        case MORTALITY:     return "MORTALITY";
        default:            return "";
        }
    }
    
    std::string filename(Stat stat)
    {
        return FILENAME_PREFIX + to_string(stat) + FILENAME_SUFFIX;
    }
    
    /// ! deallocates memory
    template<class T>
    static void debilitate(T* ptr)
    {
        if (ptr)
            delete ptr;
    }
    
    /// ! deallocates [] memory
    template<class T>
    static void debilitate_arr(T* ptr)
    {
        if (ptr)
            delete[] ptr;
    }
    
    template<class T>
    static inline T min(T a, T b)
    {
        return a < b ? a : b;
    }
    
    static unsigned count_lines(const char* filename)
    {
        std::ifstream ifs{filename};
        std::string buffer;
        unsigned count = 0;
        while (std::getline(ifs, buffer))
            count++;
        return count;
    }
    
    /**
     * @brief   Iterates a line.
     * @note    Reuse from PA3.
     */
    class CSVLineYumYums
    {
        const char* m_line = nullptr;
        int m_length = 0;
        int m_index = 0;
        
    public:
        /// constructors:
        CSVLineYumYums(const char* line, int length = -1)
        {
            if (length == -1)
                length = int(strlen(line));
            
            m_line = line;
            m_length = length;
            m_index = 0;
        }
        
        CSVLineYumYums(CSVLineYumYums const&) = delete;
        
        /// assignment:
        CSVLineYumYums& operator= (CSVLineYumYums const&) = delete;
        
        
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
            debilitate_arr(str);
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
}


//******** IMPLEMENTATION OF DayStat *******

DayStat::DayStat() : DayStat{0, 0}
{
}

DayStat::DayStat(int _cases, int _deaths) : cases(_cases), deaths(_deaths)
{
}

/**
 * @param   d Raw data
 * @param   denominator Value representing an unit such as area or population
 */
DayStat::DayStat(const DayStat &d, double denominator)
    : cases{denominator == 0.0 ? 0.0 : d.cases / denominator}
    , deaths{denominator == 0.0 ? 0.0 : d.deaths / denominator}
{
}

/**
 * @return  Mortality (percentage of cases that resulted in death).
 */
double DayStat::mortalityRate() const
{
    return cases == 0.0 ? 0.0 : Util::min(100.0 * deaths / cases, 100.0);
}

double DayStat::getcases() const
{
    return cases;
}

double DayStat::getdeaths() const
{
    return deaths;
}

//******** IMPLEMENTATION OF Region *******
Region::Region()
    : raw{nullptr}
    , name{nullptr}
    , population{0}
    , area{0}
    , nday{0}
    , normPop{nullptr}
    , normArea{nullptr}
    , mortality{nullptr}
{
}

Region::~Region()
{
    Util::debilitate_arr(raw);
    Util::debilitate_arr(name);
    Util::debilitate_arr(normPop);
    Util::debilitate_arr(normArea);
    Util::debilitate_arr(mortality);
}

/**
 * @param   line A char array, a single line read from the csv file
 */
void Region::readline(char *line)
{
    //  CSV format follows:
    //  name,population,area,[cases_i,deaths_i,...]
    int len = static_cast<int>(strlen(line));
    Util::CSVLineYumYums yumyums{line, len};
    name = yumyums.next();
    population = yumyums.next_int();
    area = yumyums.next_int();
    
    if (raw)
    {
        std::cout << "Note: recreating raw array... Something went wrong..." << std::endl;
        return;
    }
    
    nday = (Util::CSVLineYumYums{line, len}.fields() - 3) / 2;
    raw = new DayStat[nday];
    for (int i = 0; i < nday; ++i)
    {
        int cases = yumyums.next_int();
        int deaths = yumyums.next_int();
        raw[i] = DayStat(cases, deaths);
    }
}

/**
 * @brief   Normalises stats by population per 1,000,000 people
 */
void Region::normalizeByPopulation()
{
    Util::debilitate_arr(normPop);
    
    normPop = new DayStat[nday];
    for (int i = 0; i < nday; ++i)
        normPop[i] = DayStat(raw[i], (population / 1000000.0));
}

/**
 * @brief   Normalises stats by area per 1,000 km^2
 */
void Region::normalizeByArea()
{
    Util::debilitate_arr(normArea);
    
    normArea = new DayStat[nday];
    for (int i = 0; i < nday; ++i)
        normArea[i] = DayStat(raw[i], (area / 1000.0));
}

void Region::computeMortalityRate()
{
    Util::debilitate_arr(mortality);
    
    mortality = new double[nday];
    for (int i = 0; i < nday; ++i)
        mortality[i] = raw[i].mortalityRate();
}

/**
 * @param   stat An element of the Enum Stat and indicates which kind of data need to be stored in csv files.
 *          See definition of Stat. As you need to generate 7 csv files, this function will be called 7 times
 *          for each region in writecsvs().
 */
void Region::write(Stat stat) const
{
    std::ofstream ofs{Util::filename(stat), std::ios::out | std::ios::app};
    ofs << std::fixed << std::setprecision(6);
    ofs << name;
    for (int i = 0; i < nday; ++i)
    {
        ofs << ",";
        switch (stat)
        {
        case CASESRAW:      ofs << raw[i].getcases(); break;
        case DEATHSRAW:     ofs << raw[i].getdeaths(); break;
        case CASESPOP:      ofs << normPop[i].getcases(); break;
        case DEATHSPOP:     ofs << normPop[i].getdeaths(); break;
        case CASESAREA:     ofs << normArea[i].getcases(); break;
        case DEATHSAREA:    ofs << normArea[i].getdeaths(); break;
        case MORTALITY:     ofs << mortality[i]; break;
        }
    }
    ofs << std::endl;
}

//******** IMPLEMENTATION OF FILE I/O FUNCTIONS *******

/**
 * @param   region An array of regions. Each element stores the information of a country (or region)
 * @param   csvFileName Path to the csv file
 * @return  length of region array
 */
int readcsv(Region*& region, const char* csvFileName)
{
    int num_regions = Util::count_lines(csvFileName) - 1;
    std::ifstream ifs{csvFileName};
    if (!ifs)
    {
        std::cout << "input file not found: " << csvFileName << std::endl;
        return 0;
    }
    region = new Region[num_regions];
    
    char line[2048];
    ifs.getline(line, 2048);    //  read header line
    for (int i = 0; ifs.getline(line, 2048); ++i)
        region[i].readline(line);
    
    return num_regions;
}

/**
 * @param   region An array of regions. Each element stores the information of a country (or region).
 * @param   nRegions The length of the region array.
 */
void writecsvs(const Region* region, int nRegions)
{
    FOREACH_STAT(stat)
    {
        //  empty files and check
        std::string filename = Util::filename(stat);
        std::ofstream ofs{filename};
        if (!ofs)
        {
            std::cout << "Unable to open file " << filename << "... aborting..." << std::endl;
            return;
        }
    }
    
    for (int i = 0; i < nRegions; ++i)
    {
        FOREACH_STAT(stat)
        {
            region[i].write(stat);
        }
    }
}


