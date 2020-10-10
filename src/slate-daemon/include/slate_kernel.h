#pragma once
class SlateKernel;

#include <iostream>
#include <fstream>
#include <string>
#include <cctype>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <fstream>
#include <nvrtc.h>
#include "sched.h" 
#include "safecuda.h"
#include "except.h"
#include "slate_sched.h"
#include "kernel_scan.h"

class SlateKernel
{
    public:

        SlateKernel(const char* codepath, const char* kname, dim3 bs, dim3 ts, size_t smem, cudaStream_t river);
        SlateKernel() = delete;
        SlateKernel(SlateKernel&) = delete;
        SlateKernel& operator=(SlateKernel&) = delete;
        ~SlateKernel();
        
        void updateBlocksAndThreads(dim3& bs, dim3& ts);
        template <typename T>
        void update_param(size_t param, T val);
        void launch(void* args[], size_t, size_t reps);
        void reschedule();


    private:
        dim3 blocks;
        dim3 threads;
        size_t smem;
        cudaStream_t stream;
        CUfunction kernel;
        const char* kernel_name;
        CUmodule module;
        
        std::vector<CUdeviceptr> sched_params;


        void compile(const char* codepath, nvrtcProgram& prog);
        void linkandload(nvrtcProgram& prog);
};


template <typename T>
void SlateKernel::update_param(size_t param, T val)
{
    if (param < sched_params.size())
    {
        printf("smval: %d\n", val);
        cuMemcpyHtoDAsync(sched_params[param], &val, sizeof(T), stream);
        CHECK_CUDA_ERROR();
    }
}
