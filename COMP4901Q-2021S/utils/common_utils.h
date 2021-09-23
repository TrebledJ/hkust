#ifndef COMMON_UTILTIES_H
#define COMMON_UTILTIES_H

#include "config.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>


#define CONSOLE_WIDTH       80
#define TIMER_RESULT_INDENT 8


namespace Utils
{
    /// @brief  Displays a header marking the beginning of a timer result.
    void timer_header(const std::string& name)
    {
        const std::string eqs((CONSOLE_WIDTH - 4 - 5 - 5 - name.size()) / 2, '=');
        std::cout << " @" << eqs + (name.size() % 2 ? "=" : "") << " Timer <" << name << "> " << eqs << "@ "
                  << std::endl;
    }

    /// @brief  Displays a header marking the beginning of a timer result.
    void compare_header(const std::string& name, const std::string& name2)
    {
        const std::string bias(5, '=');
        const std::string eqs1(CONSOLE_WIDTH - 4 - 4 - name.size() - bias.size(), '=');
        std::cout << " /" << bias << " <" << name << "> " << eqs1 << "/ " << std::endl;

        const std::string eqs2(CONSOLE_WIDTH - 4 - 3 - 5 - name2.size() - bias.size(), '=');
        std::cout << " /" << eqs2 << " vs. <" << name2 << "> " << bias << "/ " << std::endl;
    }


    /// Sugar.
    using time_point = std::chrono::high_resolution_clock::time_point;
    time_point now() { return std::chrono::high_resolution_clock::now(); }


    namespace Timing
    {
        /**
         * @brief   Gathers and accumulates the results of various timers.
         */
        class TimerResult
        {
            static unsigned counter;
            const std::string name;
            std::vector<std::chrono::nanoseconds> times;

            bool silent = false;
            bool shown = false;
            bool null = false;

            std::chrono::nanoseconds m_sum{0};
            std::chrono::nanoseconds m_min{std::chrono::hours{100}};
            std::chrono::nanoseconds m_max{0};

        public:
            TimerResult() : null{true} {}
            TimerResult(std::string name) : name{!name.empty() ? name : std::to_string(++counter)} {}

            ~TimerResult()
            {
                if (!shown)
                    show();
            }

            void push(std::chrono::nanoseconds time)
            {
                if (null)
                    return;

                times.push_back(time);
                m_sum += time;
                if (time < m_min)
                    m_min = time;
                if (time > m_max)
                    m_max = time;
            }

            void hush() { silent = true; }

            void show()
            {
                if (silent || null)
                    return;

                shown = true;

                // Show results.
                const std::string indent(TIMER_RESULT_INDENT, ' ');
                timer_header(name);

                if (times.empty())
                {
                    // Nothing to compute.
                    std::cout << indent << "(No time data.)\n" << std::endl;
                    return;
                }

                // Calculate basic statistics.
                const auto avg_ = avg();

                // Scale up the unit based on time value of avg. Use nanoseconds if tiny, seconds if large.
                std::string unit;
                float factor;
                if (avg_ < std::chrono::nanoseconds(500))
                    unit = "ns", factor = 1;
                else if (avg_ < std::chrono::microseconds(500))
                    unit = "us", factor = 1e3;
                else if (avg_ < std::chrono::milliseconds(500))
                    unit = "ms", factor = 1e6;
                else
                    unit = "s", factor = 1e9;

                const auto real_avg = avg_.count() / factor;
                const auto real_std = stddev().count() / factor;
                const auto real_min = min().count() / factor;
                const auto real_max = max().count() / factor;

                int width = longest({real_avg, real_std, real_min, real_max});

                std::cout << std::setprecision(3) << std::fixed;
                std::cout << indent << "avg: " << std::setw(width) << real_avg << unit << "\n";
                std::cout << indent.substr(0, indent.size() - 3) << "stddev: " << std::setw(width) << real_std << unit
                          << "\n";
                std::cout << indent << "min: " << std::setw(width) << real_min << unit << "\n";
                std::cout << indent << "max: " << std::setw(width) << real_max << unit << "\n";
                std::cout << std::endl;
            }

            void compare(const TimerResult& other) const
            {
                if (null)
                    return;

                const std::string indent(TIMER_RESULT_INDENT, ' ');

                std::cout << std::setprecision(3) << std::fixed;
                compare_header(name, other.name);

                const auto spdup_avg = 1.0 * avg().count() / other.avg().count();
                const auto spdup_min = 1.0 * min().count() / other.max().count();
                const auto spdup_max = 1.0 * max().count() / other.min().count();

                int width = longest({spdup_avg, spdup_min, spdup_max});

                std::cout << indent.substr(0, indent.size() - 2) << "Speedup:\n";
                std::cout << indent << "avg: " << std::setw(width) << spdup_avg << " (" << (spdup_avg > 1.0 ? '+' : '-')
                          << ")\n";
                std::cout << indent << "min: " << std::setw(width) << spdup_min << "\n";
                std::cout << indent << "max: " << std::setw(width) << spdup_max << "\n";
                std::cout << std::endl;
            }

            std::chrono::nanoseconds avg() const
            {
                return times.empty() ? std::chrono::nanoseconds(0) : std::chrono::nanoseconds(m_sum / times.size());
            }
            std::chrono::nanoseconds min() const { return m_min; }
            std::chrono::nanoseconds max() const { return m_max; }
            std::chrono::nanoseconds stddev() const
            {
                const int64_t mean = avg().count();
                uint64_t sum = 0;
                for (const auto& t : times)
                {
                    int64_t x = t.count() - mean;
                    sum += x * x;
                }
                return std::chrono::nanoseconds(uint64_t(sqrt(sum / times.size())));
            }

            static TimerResult& instance()
            {
                static TimerResult base{"Global"};
                return base;
            }

        private:
            static int longest(std::vector<double> v)
            {
                int max = 0;
                for (const auto& f : v)
                {
                    std::stringstream ss;
                    ss << std::setprecision(3) << std::fixed << f;
                    int len = ss.str().size();
                    if (len > max)
                        max = len;
                }
                return max;
            }
        };

        unsigned TimerResult::counter = 0;


        /**
         * @brief   A simple-to-use RAII-based class for timing work within a scope.
         *          Use in conjunction with TimerResult.
         *
         * @note    This class introduces approximately 50ns (1e-9 second) overhead.
         *          This can be eliminated by calling the `eliminate_overhead()` method.
         */
        class Timer
        {
            TimerResult* res = nullptr;
            time_point start;

        public:
            Timer(TimerResult* res = nullptr) : res{res} { start = now(); }

            ~Timer()
            {
                // Record the time.
                const auto end = now();
                const auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
                if (res)
                    res->push(time);
                else
                    TimerResult::instance().push(time);
            }
        };

    } // namespace Timing


    /**
     * @brief   A good quality RNG for floats using std::mt19937 as a generator.
     */
    struct randfloat
    {
        float low, high;

        randfloat(float low = 0.0, float high = 1.0) : low{low}, high{high} {}

        float operator()()
        {
            static std::mt19937 gen{std::random_device{}()};
            std::uniform_real_distribution<float> dis{low, high};
            return dis(gen);
        }
    };


    namespace IO
    {
        namespace detail
        {
            bool bad_read(const std::string& reason = "")
            {
                std::cin.clear();
                std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                if (reason.empty())
                    std::cout << "Input rejected.\n" << std::endl;
                else
                    std::cout << "Input rejected: " << reason << ".\n" << std::endl;
                return true;
            }

            template <typename T>
            bool read(T& t)
            {
                if (!(std::cin >> t))
                {
                    bad_read("parse failed");
                    return false;
                }
                return true;
            }

            template <typename T>
            bool input_impl(T& x)
            {
                return read(x);
            }

            template <typename T, typename... Ts>
            bool input_impl(T& x, Ts&... xs)
            {
                return read(x) && input_impl(xs...);
            }
        } // namespace detail

        template <typename... Ts>
        void input(std::string prompt, Ts&... xs)
        {
            do
            {
                std::cout << prompt;
            } while (!detail::input_impl(xs...));
        }

        /**
         * @brief   Input function that accepts a validator. The validator's signature should be
         *          bool(string&, const Ts&...), where Ts... are the types of the input arguments.
         */
        template <typename Func, typename... Ts>
        void inputv(std::string prompt, Func validator, Ts&... xs)
        {
            std::string reason;
            do
            {
                std::cout << prompt;
            } while (!detail::input_impl(xs...) || (!validator(reason, xs...) && detail::bad_read(reason)));
        }

// Helper macros for writing input validators.
#define validator(...) [&](std::string & reason, __VA_ARGS__)
#define require(condition, reason_) \
    do                              \
    {                               \
        if (!(condition))           \
        {                           \
            reason = reason_;       \
            return false;           \
        }                           \
    } while (0)

#define positive                                    \
    validator(int64_t n)                            \
    {                                               \
        require(n > 0, "input should be positive"); \
        return true;                                \
    }

#define non_negative                                     \
    validator(int64_t n)                                 \
    {                                                    \
        require(n >= 0, "input should be non-negative"); \
        return true;                                     \
    }

    } // namespace IO
} // namespace Utils


#endif