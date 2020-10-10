__global__ void clock_block(clock_t* d_o, long clock_count)
{
    clock_t start_clock = clock64();
    volatile long clock_offset = 0;
    volatile int i = 0;
    for (i = 0; i < 10000000; i++)
    while (clock_offset < clock_count)
    {
    clock_offset = clock_count--;
    }
    d_o[0] = clock_offset;
}
