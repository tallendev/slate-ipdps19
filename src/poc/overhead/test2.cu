#include <iostream>
#include <cuda.h>

__global__ void gloob()
{
    return;
}

__global__ void glob()
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
       gloob<<<13, 128>>>(); 
    }
    return;
}

int main()
{
    float time;
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);  
    cudaEventRecord(start, 0);
    glob<<<13, 128>>>();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);  
    cudaEventElapsedTime(&time, start, stop);
    std::cout << time << std::endl;
    return 0;
}
