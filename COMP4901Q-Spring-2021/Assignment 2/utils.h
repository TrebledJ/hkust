#ifndef UTILTIES_H
#define UTILTIES_H

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <vector>


/// @brief  Displays a header marking the beginning of a timer result.
void timer_header(const std::string& name)
{
    const std::string eqs((CONSOLE_WIDTH - 4 - 5 - 5 - name.size()) / 2, '=');
    std::cout << " @" << eqs + (name.size() % 2 ? "=" : "") << " Timer <" << name << "> " << eqs << "@ " << std::endl;
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
time_point now()
{
    return std::chrono::high_resolution_clock::now();
}


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

        std::chrono::nanoseconds m_sum{0};
        std::chrono::nanoseconds m_min{std::chrono::hours{100}};
        std::chrono::nanoseconds m_max{0};

    public:
        TimerResult(std::string name = "") : name{!name.empty() ? name : std::to_string(++counter)} {}

        ~TimerResult()
        {
            if (!shown)
                show();
        }

        void push(std::chrono::nanoseconds time)
        {
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
            if (silent)
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
            const auto stddev_ = stddev();
            const auto min_ = min();
            const auto max_ = max();

            // Scale up the unit based on time value of avg. Use nanoseconds if tiny, seconds if large.
            std::string unit;
            float factor;
            if (avg_ < std::chrono::nanoseconds(100))
                unit = "ns", factor = 1;
            else if (avg_ < std::chrono::microseconds(100))
                unit = "us", factor = 1e3;
            else if (avg_ < std::chrono::milliseconds(500))
                unit = "ms", factor = 1e6;
            else
                unit = "s", factor = 1e9;

            std::cout << std::setprecision(3) << std::fixed;
            std::cout << indent << "avg: " << avg_.count() / factor << unit << "\n";
            std::cout << indent.substr(0, indent.size() - 3) << "stddev: " << stddev_.count() / factor << unit << "\n";
            std::cout << indent << "min: " << min_.count() / factor << unit << "\n";
            std::cout << indent << "max: " << max_.count() / factor << unit << "\n";
            std::cout << std::endl;
        }

        void compare(const TimerResult& other)
        {
            const std::string indent(TIMER_RESULT_INDENT, ' ');

            std::cout << std::setprecision(3) << std::fixed;
            compare_header(name, other.name);

            const auto spdup_avg = 1.0 * avg().count() / other.avg().count();
            const auto spdup_min = 1.0 * min().count() / other.max().count();
            const auto spdup_max = 1.0 * max().count() / other.min().count();

            std::cout << indent.substr(0, indent.size() - 2) << "Speedup:\n";
            std::cout << indent << "avg: " << spdup_avg << " (" << (spdup_avg > 1.0 ? '+' : '-') << ")\n";
            std::cout << indent << "min: " << spdup_min << "\n";
            std::cout << indent << "max: " << spdup_max << "\n";
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

#if ENABLE_CUDA
    class DeviceTimer
    {
        TimerResult* res = nullptr;
        cudaEvent_t start, stop;

    public:
        DeviceTimer(TimerResult* res = nullptr) : res{res}
        {
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start, 0);
        }

        ~DeviceTimer()
        {
            // Stop timer.
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaDeviceSynchronize();

            float ms;
            cudaEventElapsedTime(&ms, start, stop);

            // Record the time.
            const auto time = std::chrono::nanoseconds(static_cast<uint32_t>(ms * 1e6));
            if (res)
                res->push(time);
            else
                TimerResult::instance().push(time);
        }
    };
#endif
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


namespace Input
{
    bool bad_read()
    {
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        std::cout << "Input rejected.\n" << std::endl;
        return true;
    }

    template <typename T>
    bool read(T& t)
    {
        if (!(std::cin >> t))
        {
            bad_read();
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

    template <typename Func, typename... Ts>
    void input(const char* prompt, Ts&... xs)
    {
        do
        {
            std::cout << prompt;
        } while (!input_impl(xs...));
    }

    template <typename Func, typename... Ts>
    void input(const char* prompt, Func validator, Ts&... xs)
    {
        do
        {
            std::cout << prompt;
        } while (!input_impl(xs...) || (validator && !validator(xs...) && bad_read()));
    }
} // namespace Input


namespace Memory
{

#ifdef ENABLE_CUDA

// Macro to check for errors on CUDA API. Adapted from tutorial notes.
#define CHECK(call)                                                          \
    {                                                                        \
        const cudaError_t error = call;                                      \
        if (error != cudaSuccess)                                            \
        {                                                                    \
            printf("Error on %s:%d, code: %d\n", __FILE__, __LINE__, error); \
            printf("Reason: %s\n", cudaGetErrorString(error));               \
            exit(1);                                                         \
        }                                                                    \
    }

    template <typename T>
    using device_ptr = std::unique_ptr<T, decltype(&cudaFree)>;

    template <typename T>
    class device_array
    {
        device_ptr<T> ptr;
        size_t size;

    public:
        device_array(uint32_t len, T init = T{}) : ptr{nullptr, cudaFree}, size{len}
        {
            T* p;
            CHECK(cudaMalloc((void**)&p, len * sizeof(T)));
            CHECK(cudaMemset(p, init, len * sizeof(T)));
            ptr = device_ptr<T>(p, cudaFree);
        }

        device_array(const T* copyfrom, uint32_t len) : ptr{nullptr, cudaFree}, size{len}
        {
            T* p;
            CHECK(cudaMalloc((void**)&p, len * sizeof(T)));
            CHECK(cudaMemcpy(p, copyfrom, len * sizeof(T), cudaMemcpyHostToDevice));
            ptr = device_ptr<T>(p, cudaFree);
        }

        device_array(const device_array&) = delete;
        device_array& operator=(const device_array&) = delete;

        device_array(device_array&&) = default;
        device_array& operator=(device_array&&) = default;

        void copy_to(T* dest, cudaMemcpyKind kind = cudaMemcpyDeviceToHost)
        {
            CHECK(cudaMemcpy(dest, ptr.get(), size * sizeof(T), kind));
        }

        operator T*() const { return ptr.get(); }
    };

#endif

} // namespace Memory


using namespace Timing;
using namespace Input;
using namespace Memory;


#endif