__device__ inline uint3 operator+(const dim3& a, const dim3& b)
{
    uint3 val = {a.x + b.x, a.y + b.y, a.z + b.z};   
    return val;
}

__device__ inline uint3 operator+(const uint3& a, const uint3& b)
{
    uint3 val = {a.x + b.x, a.y + b.y, a.z + b.z};   
    return val;
}

extern "C"
__global__ void test(void)
{
    if (threadIdx.x == 0)
    {
        printf ("blockIdx: %d, %d\n", blockIdx.x, blockIdx.y);
    }
}

extern "C"
__global__ void goob(dim3 gshift, uint3 bshift, char* oh)
{
}
/*
extern "C"
__global__ void goobold()
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
        printf("Noice!\n");
    return;
}
*/
