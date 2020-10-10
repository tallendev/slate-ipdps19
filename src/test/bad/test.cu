
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
