extern "C"
__global__ void test( int testeroo)
{
    if (threadIdx.x == 0)
    {
        printf ("blockIdx: %d, %d\n", blockIdx.x, blockIdx.y);
    }
}

