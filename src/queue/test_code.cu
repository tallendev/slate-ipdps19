extern "C"
__global__ void test1(int* test)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        printf("hello from kernel\n");
        do_stuff();//inc_val();
        test[0] = get_val();
    }
}

