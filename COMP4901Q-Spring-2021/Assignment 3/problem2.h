#ifndef PROBLEM2_H
#define PROBLEM2_H

#include "context.h"
#include "matrix.h"
#include "utils.h"

#include <cassert>


#define BENCH_FUNCTION_2(func) Utils::Timing::TimerResult func(const ContextP2& ctx, float& output)


template <typename T>
T operate(ContextP2::Operation op, T a, T b)
{
    switch (op)
    {
    case ContextP2::SUM: return a + b;
    case ContextP2::MAX: return std::max(a, b);
    default: return T{};
    }
}

template <typename T>
T operate(MPI_Op op, T a, T b)
{
    if (op == MPI_SUM)
        return a + b;
    if (op == MPI_MAX)
        return std::max(a, b);
    return T{};
}


BENCH_FUNCTION_2(serial_reduce)
{
    using namespace Utils::Timing;

    TimerResult timings{"Array Reduction: Serial/CPU"};

    for (int i = 0; i < ctx.num_runs; i++)
    {
        Timer timer{&timings};

        // Do the thing!
        if (ctx.op == ContextP2::SUM)
        {
            output = 0.0;
            for (int i = 0; i < ctx.array.size(); i++)
                output += ctx.array[i];
        }
        else if (ctx.op == ContextP2::MAX)
        {
            output = -9e99;
            for (int i = 0; i < ctx.array.size(); i++)
                if (ctx.array[i] > output)
                    output = ctx.array[i];
        }
    }

    timings.show();

    if (ctx.print_output)
    {
        std::cout << "  Output: " << output << "\n" << std::endl;
    }

    return timings;
}


#if ENABLE_MPI
float parallel_allreduce_mpi_impl(const ContextP2& ctx)
{
    // Very simple, we'll first scatter the data. Each process does their own computation, then the
    // data is collated using Allreduce.

    Vector local_arr{ctx.n / ctx.num_procs};

    // Distribute the array.
    CHECK(MPI_Scatter(ctx.array.data(), local_arr.size(), MPI_FLOAT, local_arr.data(), local_arr.size(), MPI_FLOAT, 0,
                      MPI_COMM_WORLD));

    // Do the thing!
    float local_res;
    if (ctx.op == ContextP2::SUM)
    {
        local_res = 0.0;
        for (int i = 0; i < local_arr.size(); i++)
            local_res += local_arr[i];
    }
    else
    {
        local_res = -9e99;
        for (int i = 0; i < local_arr.size(); i++)
            if (local_arr[i] > local_res)
                local_res = local_arr[i];
    }

    // Aggregate the results.
    const MPI_Op op = (ctx.op == ContextP2::SUM ? MPI_SUM : ctx.op == ContextP2::MAX ? MPI_MAX : MPI_OP_NULL);

    float res = 0.0;
    CHECK(MPI_Allreduce(&local_res, &res, 1, MPI_FLOAT, op, MPI_COMM_WORLD));

    return res;
}
#endif


BENCH_FUNCTION_2(parallel_allreduce_mpi)
{
    using namespace Utils::Timing;

#if ENABLE_MPI
    using namespace Utils::MPI;

    if (ctx.mpi_id == MASTER)
    {
        TimerResult timings{"Array Reduction: Parallel/MPI"};

        for (int i = 0; i < ctx.num_runs; i++)
        {
            CHECK(MPI_Barrier(MPI_COMM_WORLD)); // Sync processes.

            MPITimer timer{&timings};
            output = parallel_allreduce_mpi_impl(ctx);
        }

        timings.show();

        if (ctx.print_output)
        {
            std::cout << "  Output: " << output << "\n" << std::endl;
        }

        return timings;
    }
    else
    {
        for (int i = 0; i < ctx.num_runs; i++)
        {
            CHECK(MPI_Barrier(MPI_COMM_WORLD));
            parallel_allreduce_mpi_impl(ctx);
        }
    }

#endif

    return TimerResult{};
}


#if ENABLE_MPI
template <typename T>
int RING_Allreduce(const T* sendbuf, T* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
{
    assert((op == MPI_SUM || op == MPI_MAX) && "Operation not supported.");
    assert(sendbuf && "sendbuf cannot be null");
    assert(recvbuf && "recvbuf cannot be null");

    int id, num_procs;
    CHECK(MPI_Comm_rank(comm, &id));
    CHECK(MPI_Comm_size(comm, &num_procs));
    assert(count <= num_procs); // TODO: Assume for now that count == num_procs.

    const int dest_id = (id + 1 == num_procs ? 0 : id + 1);
    const int src_id = (id == 0 ? num_procs - 1 : id - 1);

    // Copy the send buffer into the recv buffer.
    // From now on, we'll only need the recv buffer.
    for (int i = 0; i < count; i++)
        recvbuf[i] = sendbuf[i];

    MPI_Request send_req, recv_req;
    MPI_Status status;

    // Need `N - 1` iterations, since we're adding `N` terms.
    for (int i = 0; i < num_procs - 1; i++)
    {
        const int idx = (id + num_procs - i) % num_procs; // Offset the index by id. We'll count backwards.
        const int prev = (idx == 0 ? num_procs - 1 : idx - 1);

        const bool no_send = (idx >= count);
        const bool no_recv = (prev >= count);

        // Cycle messages.
        T buf;
        if (!no_send)
            CHECK(MPI_Isend(&recvbuf[idx], 1, datatype, dest_id, 0, comm, &send_req));
        if (!no_recv)
            CHECK(
                MPI_Irecv(&buf, 1, datatype, src_id, 0, comm,
                          &recv_req)); // Receive data from the (id-1)th process, to be updated with the (i-1)th value.

        if (!no_send)
            CHECK(MPI_Wait(&send_req, &status));
        if (!no_recv)
            CHECK(MPI_Wait(&recv_req, &status));

        if (!no_recv)
            recvbuf[prev] = operate(op, recvbuf[prev], buf);
    }

    MPI_Barrier(comm);

    // Propagate. Update all recvbufs.
    // Last updated record (the finalised one) is at idx = id + 1. We want to propagate these records to the rest.
    for (int i = 0; i < num_procs - 1; i++)
    {
        const int idx = (id + num_procs - i + 1) % num_procs; // Offset the index by id. We'll count backwards.
        const int prev = (idx == 0 ? num_procs - 1 : idx - 1);

        const bool no_send = (idx >= count);
        const bool no_recv = (prev >= count);

        // Cycle messages.
        if (!no_send)
            CHECK(MPI_Isend(&recvbuf[idx], 1, datatype, dest_id, 0, comm, &send_req));
        if (!no_recv)
            CHECK(MPI_Irecv(&recvbuf[prev], 1, datatype, src_id, 0, comm, &recv_req)); // Receive directly into buffer.

        if (!no_send)
            CHECK(MPI_Wait(&send_req, &status));
        if (!no_recv)
            CHECK(MPI_Wait(&recv_req, &status));
    }

    return MPI_SUCCESS;
}


float parallel_allreduce_ring_impl(const ContextP2& ctx)
{
    uint32_t count = ctx.n / ctx.num_procs;
    Vector local_array{count};
    Vector crossthread_result{count};

    CHECK(MPI_Scatter(ctx.array.data(), count, MPI_FLOAT, local_array.data(), count, MPI_FLOAT, 0, MPI_COMM_WORLD));

    if (count > ctx.num_procs)
    {
        // Aggregate until count == ctx.num_procs.
        float res = 0;
        for (int i = ctx.num_procs; i < count; i++)
            res = operate(ctx.op, res, local_array[i]);

        local_array[0] = operate(ctx.op, local_array[0], res);
        count = ctx.num_procs;
    }

    MPI_Op op = (ctx.op == ContextP2::SUM ? MPI_SUM : ctx.op == ContextP2::MAX ? MPI_MAX : MPI_OP_NULL);
    RING_Allreduce(local_array.data(), crossthread_result.data(), count, MPI_FLOAT, op, MPI_COMM_WORLD);

    // Reduce the final result.
    if (ctx.mpi_id == MASTER)
    {
        float final_res = 0;
        for (int i = 0; i < count; i++)
            final_res = operate(ctx.op, final_res, crossthread_result[i]);
        return final_res;
    }
    return 0;
}
#endif


BENCH_FUNCTION_2(parallel_allreduce_ring)
{
    using namespace Utils::Timing;

#if ENABLE_MPI
    using namespace Utils::MPI;

    if (ctx.mpi_id == MASTER)
    {
        TimerResult timings{"Array Reduction: Parallel/Ring"};

        for (int i = 0; i < ctx.num_runs; i++)
        {
            CHECK(MPI_Barrier(MPI_COMM_WORLD)); // Sync processes.

            MPITimer timer{&timings};
            output = parallel_allreduce_ring_impl(ctx);
        }

        timings.show();

        if (ctx.print_output)
        {
            std::cout << "  Output: " << output << "\n" << std::endl;
        }

        return timings;
    }
    else
    {
        for (int i = 0; i < ctx.num_runs; i++)
        {
            CHECK(MPI_Barrier(MPI_COMM_WORLD));
            parallel_allreduce_ring_impl(ctx);
        }
    }

#endif

    return TimerResult{};
}


#endif