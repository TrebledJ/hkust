#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H


#include "common_utils.h"

#include <chrono>
#include <memory>


#if ENABLE_CUDA

#include <cuda.h>


// Macro to check for errors on CUDA API. Adapted from tutorial notes.
#define CUDA_CHECK(call)                                                     \
    {                                                                        \
        const cudaError_t error = call;                                      \
        if (error != cudaSuccess)                                            \
        {                                                                    \
            printf("Error on %s:%d, code: %d\n", __FILE__, __LINE__, error); \
            printf("Reason: %s\n", cudaGetErrorString(error));               \
            exit(1);                                                         \
        }                                                                    \
    }

namespace Utils
{
    namespace CUDA
    {
        class DeviceTimer
        {
            Timing::TimerResult* res = nullptr;
            cudaEvent_t start, stop;

        public:
            DeviceTimer(Timing::TimerResult* res = nullptr) : res{res}
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
                    Timing::TimerResult::instance().push(time);
            }
        };


        template <typename T>
        using DevicePointer = std::unique_ptr<T, decltype(&cudaFree)>;

        template <typename T>
        class DeviceArray
        {
            DevicePointer<T> ptr;
            size_t m_size;

        public:
            DeviceArray(uint32_t len, T init = T{}) : ptr{nullptr, cudaFree}, m_size{len}
            {
                T* p;
                CUDA_CHECK(cudaMalloc((void**)&p, len * sizeof(T)));
                CUDA_CHECK(cudaMemset(p, init, len * sizeof(T)));
                ptr = DevicePointer<T>(p, cudaFree);
            }

            DeviceArray(const T* copyfrom, uint32_t len) : ptr{nullptr, cudaFree}, m_size{len}
            {
                T* p;
                CUDA_CHECK(cudaMalloc((void**)&p, len * sizeof(T)));
                CUDA_CHECK(cudaMemcpy(p, copyfrom, len * sizeof(T), cudaMemcpyHostToDevice));
                ptr = DevicePointer<T>(p, cudaFree);
            }

            DeviceArray(const DeviceArray&) = delete;
            DeviceArray& operator=(const DeviceArray&) = delete;

            DeviceArray(DeviceArray&&) = default;
            DeviceArray& operator=(DeviceArray&&) = default;

            void copy_to(T* dest, cudaMemcpyKind kind = cudaMemcpyDeviceToHost)
            {
                CUDA_CHECK(cudaMemcpy(dest, ptr.get(), m_size * sizeof(T), kind));
            }

            T* get() { return ptr.get(); }
            const T* get() const { return ptr.get(); }
            size_t size() const { return m_size; }

            operator T*() const { return ptr.get(); }
        };

    } // namespace CUDA

} // namespace Utils


#endif

#endif