#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H


#include "common_utils.h"


#if ENABLE_CUDA

namespace Utils
{

    namespace Timing
    {
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
    } // namespace Timing


    namespace CUDA
    {

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

    } // namespace CUDA

} // namespace Utils


#endif

#endif