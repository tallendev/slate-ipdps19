__device__ int val = 5;

__device__ void inc_val()
{
    val += 1;
}

__device__ void do_stuff()
{
    volatile void* shit = (void*) inc_val;
    void (*f)() = (void (*)())shit;
    f();
}

__device__ int get_val()
{
    return val;
}
