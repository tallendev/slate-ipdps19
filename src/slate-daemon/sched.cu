
typedef unsigned int uint;
#define WARP_SZ 32
__device__ inline int lane_id(void) { return threadIdx.x % WARP_SZ; }
__device__ __inline__ uint get_smid(void)
{
    uint ret;
    asm("mov.u32 %0, %smid;" : "=r"(ret) );
    return ret;
}
    __device__ volatile int retreat = 0;
__device__ unsigned int slate_idx = 0;
//__device__ volatile int living_blocks = 0;
//__device__ const int slate_blocksize = 256;
__device__ const unsigned int slate_max = 409600;
__device__ const uint3 gridDimm = {640, 640, 1};
#define TILE_DIM    16
#define BLOCK_ROWS  16

#define FLOOR(a,b) (a-(a%b))

__global__ void transposeNaive(const uint sm_low, const uint sm_high, float* odata, float* idata, int width, int height)
{
    #define SLATE_ITERS 10 
            __shared__ unsigned int id;
    __shared__ uint valid_task;
    uint slate_smid;
    const int leader = (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0);
    if (leader)
    {
        id = 0;
        slate_smid = get_smid();
        valid_task = slate_smid < sm_low || slate_smid >= sm_high;
    }
    __syncthreads();
    if (valid_task){ return;}
    __shared__ uint3 shared_blockIdxx;
    __shared__ int iters;
    uint local_id;
    do
    {
        if (leader)
        {
            local_id = atomicAdd(&slate_idx, SLATE_ITERS);
            iters = min((int)SLATE_ITERS, (int)slate_max - (int)local_id);
            id = local_id + SLATE_ITERS;
            shared_blockIdxx.x= local_id % gridDimm.x - 1;
            shared_blockIdxx.y = local_id / gridDimm.x;
        }
        __syncthreads();
//        local_id = id - 10;//deletememem
        const int local_iters = iters;
        uint3 blockIdxx;
        blockIdxx.x = shared_blockIdxx.x;
        blockIdxx.y = shared_blockIdxx.y;
        for (int slate_counter = 0; slate_counter < local_iters; ++slate_counter)
        {
            blockIdxx.x++;
            if (blockIdxx.x == gridDimm.x)
            {
                blockIdxx.x = 0;
                blockIdxx.y++;
            }
            //assert (blockIdxx.x < gridDimm.x);
            //assert (blockIdxx.y < gridDimm.y);
        
    int xIndex = blockIdxx.x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdxx.y * TILE_DIM + threadIdx.y;

    int index_in  = xIndex + width * yIndex;
    int index_out = yIndex + height * xIndex;

    for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS)
    {
        odata[index_out+i] = idata[index_in+i*width];
    }
        }
    } while(!retreat && id < slate_max);
}
//const __device__ uint3 grid = {640, 640, 1};
const __device__ uint3 grid = {240, 1, 1};
const __device__ uint3 block = {16, 16, 1};
extern "C" __global__ void transposeNaivescheduler(int reps, volatile uint* start_sm, volatile uint* end_sm, float* odata, float* idata, int width, int height) {
    cudaStream_t s;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    retreat = 0;
    //printf("sm hi, low: %u, %u\n", *start_sm, *end_sm);
    //printf("reps: %u, %u\n", reps);
    //return;
    //for (int i = 0; i < reps; i++)
  //{
        slate_idx = 0;
        do
        {
          //  living_blocks = 0;
            transposeNaive<<<grid, block, 0, s>>>(*start_sm, *end_sm, odata, idata, width, height);
         cudaDeviceSynchronize();
         retreat = 0;
        } while (slate_idx < slate_max);
    //}
    cudaStreamDestroy(s);}

