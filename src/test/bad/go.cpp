#include <slate.h>
#include <helper_functions.h>   // helper functions for string parsing
#include <helper_cuda.h>        // helper functions CUDA error checking and initialization

const char* path = "/home/tnallen/dev/ics2017/src/test/bad/test.cu";
const char* kernel = "clock_block";

int main()
{
    clock_t* d_o;
    long clock_count = 705000;
    slateMalloc((void**)&d_o, sizeof(clock_t));
    size_t threads = 128;
    size_t blocks = 10 * 13 * 2048 / threads;

    double gpuTime;
    StopWatchInterface *hTimer = NULL;
    sdkCreateTimer(&hTimer);
    sdkResetTimer(&hTimer);
    sdkStartTimer(&hTimer);

    slateLaunchKernel(path, kernel, blocks, threads, d_o, clock_count);
    slateSync();
    slateCpyDtoH(d_o);
    sdkStopTimer(&hTimer);
    gpuTime = sdkGetTimerValue(&hTimer);
    printf("clock: %u\n", *d_o);
    printf("GPU() time    : %f msec\n", gpuTime);
    slateHangup();
}
