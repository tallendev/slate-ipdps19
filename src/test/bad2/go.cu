#include <helper_functions.h>   // helper functions for string parsing
#include <helper_cuda.h>        // helper functions CUDA error checking and initialization

const char* path = "/home/tnallen/dev/ics2017/src/test/bad/test.cu";
const char* kernel = "clock_block";


__global__ void clock_block(clock_t* d_o, volatile long clock_count)
{

    volatile long clock_offset = 0;
    long temp_clock = clock_count;
    while (clock_offset < temp_clock)
    {
        clock_offset++;
    }
    d_o[0] = clock_offset;
}
/*
__global__ void clock_block(clock_t* d_o, clock_t clock_count)
{
    clock_t start_clock = clock64();
    clock_t clock_offset = 0;
    while (clock_offset < clock_count)
    {
    clock_offset = clock64() - start_clock;
    }
    d_o[0] = clock_offset;
}
*/
int main()
{
    long d_o;
    clock_t* d_p;
    long clock_count = 705000;
    cudaMalloc((void**)&d_p, sizeof(clock_t));
    size_t threads = 128;
    size_t blocks =  10 * 13 * 2048 / threads;

    double gpuTime;
    StopWatchInterface *hTimer = NULL;
    sdkCreateTimer(&hTimer);
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);

    clock_block<<<blocks, threads>>>(d_p, clock_count);
    cudaDeviceSynchronize();
    cudaMemcpy(&d_o, d_p, sizeof(long), cudaMemcpyDeviceToHost);
    sdkStopTimer(&hTimer);
    gpuTime = sdkGetTimerValue(&hTimer);
    printf("clock: %u\n", d_o);
    printf("GPU() time    : %f msec\n", gpuTime);
}
