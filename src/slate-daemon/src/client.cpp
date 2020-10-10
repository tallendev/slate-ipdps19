#include "client.h"

#define mytimestamp() std::chrono::duration_cast< std::chrono::nanoseconds >(std::chrono::system_clock::now().time_since_epoch()).count()

typedef std::chrono::high_resolution_clock Clock;

volatile int SlateClient::current_bits = 0;
volatile int SlateClient::running_clients = 0;
volatile int SlateClient::held_clients = 0;
std::mutex SlateClient::launch_lock;
std::mutex SlateClient::query_lock;
std::condition_variable SlateClient::can_launch;

SlateClient::SlateClient(int pid, CUcontext ctx) : pid_fifo(std::to_string(pid)), context(ctx)
{
    fifo_in = ("comm/" + pid_fifo + "tos");
    fifo_out = ("comm/" + pid_fifo + "toc");
    mkfifo(fifo_in.c_str(), 0666);
    mkfifo(fifo_out.c_str(), 0666);
    fdin = open(fifo_in.c_str(), O_RDONLY);
    fdout = open(fifo_out.c_str(), O_WRONLY);
    //printf("in: %s\n", fifo_in.c_str());
    //printf("out: %s\n", fifo_out.c_str());
    // MUST SET CONTEXT BEFORE OTHER CUDA STUFF
    CUDA_SAFE_CALL(cuCtxSetCurrent(context));
    CHECK_CUDA_ERROR();
    cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING);
    CHECK_CUDA_ERROR();
    canRun = false;
    my_bits = 0;
}

bool SlateClient::operator==(const SlateClient& b)
{
    return pid_fifo == b.pid_fifo;
}
       
int SlateClient::process_cmds()
{
    SlateStatus out;
    size_t ret = 0;
    Command expect;
    int bytes = read(fdin, &expect, sizeof(Command));
    //fprintf(stderr, "expect: %u\n", expect);
    if (bytes == EOF)
    {
        ret = -1;
    }
    else
    {
        size_t reps = 1;
        switch(expect)
        {
            case cpytodevice:
                do_devcpy();
                break;
            case cpyfromdevice:
                do_hostcpy();
                break;
            case cumemset:
                do_memset();
                break;
            case cumalloc:
                do_malloc();
                break;
            case cusync:
                //fprintf(stderr, "cusync\n");
                cudaStreamSynchronize(stream);
                std::cout << "sync_start: " <<  mytimestamp() << std::endl;
                done_running();
                out = SlateStatus::sync_success;
                write(fdout, &out, sizeof(SlateStatus));
                break;
            case hangup:
                done_running();
                ret = 1;
                break;
            case cukernel:
                bytes = read(fdin, &reps, sizeof(size_t));
                try
                {
                    handle_kernel(reps);
                }
                catch (nvrtc_exception& e)
                {
                    std::cerr << e.what() << std::endl;
                    ret = -3;
                }
                break;
            
            default:
                fprintf(stderr, "Did not understand input: %d\n", expect);
                ret = -2;
        }
    }
    return ret;
}


const char* SlateClient::getpid() const
{
    return pid_fifo.c_str();
}

void SlateClient::done_running()
{
    if (canRun)
    {
        canRun = false;
        std::unique_lock<std::mutex> lock(query_lock);
        // TODO maybe problems here eventually
        if ( !(running_clients > 1 && current_bits == my_bits) )
        {
            current_bits ^= my_bits;
        }
        if (held_clients) held_clients--;
        else running_clients--;
        printf("held_clients: %d\n", held_clients);
        printf("running_clients: %d\n", running_clients);
        can_launch.notify_all();
        FairScheduler::update_sched();
    }
}

SlateClient::~SlateClient()
{
    //printf ("client decimated\n");
    cudaStreamSynchronize(stream);
    CHECK_CUDA_ERROR();
    cudaStreamDestroy(stream);
    close(fdin);
    close(fdout);
    unlink(fifo_in.c_str());
    unlink(fifo_out.c_str());
    while(!bufs.empty())
    {
        client_mem& front = bufs.front();
        shmdt (front.shm_buf); 
        shmctl (front.shm_id, IPC_RMID, 0); 
        cuMemFree(front.dev_pointer);
        //cudaFree(front.dev_pointer);
        bufs.pop_front();
    }
    done_running();
    /*
    for (auto k : kernels)  { delete k.second; }
    FairScheduler::update_sched();
    */
}

int SlateClient::do_devcpy()
{
    key_t key;
    read(fdin, &key, sizeof(key_t)); 
    struct client_mem& buf = keymap.at(key);
   // cudaMemcpy(buf.dev_pointer, buf.shm_buf, buf.size, cudaMemcpyHostToDevice);
    cuMemcpyHtoDAsync(buf.dev_pointer, buf.shm_buf, buf.size, stream);
    CHECK_CUDA_ERROR();
    return 0;
}

int SlateClient::do_hostcpy()
{
    key_t key;
    SlateStatus s = SlateStatus::read_done;
    read(fdin, &key, sizeof(key_t)); 
    struct client_mem& buf = keymap.at(key);
   // cudaMemcpy(buf.dev_pointer, buf.shm_buf, buf.size, cudaMemcpyHostToDevice);
    std::cout << "cpy2h_end: " << mytimestamp() << std::endl;
    CUDA_SAFE_CALL(cuMemcpyDtoHAsync(buf.shm_buf, buf.dev_pointer, buf.size, stream));
    std::cout << "cpy2h_start: " <<  mytimestamp() << std::endl;
    done_running();
    write(fdout, &s, sizeof(SlateStatus));
    return 0;
}

int SlateClient::do_memset()
{
    key_t key;
    size_t size;
    int val;
    read(fdin, &key, sizeof(key_t));
    read(fdin, &val, sizeof(int));
    read(fdin, &size, sizeof(size_t));
    //printf("memset key, val, size: %u %d %lu\n", key, val, size);
    struct client_mem& buf = keymap.at(key);
   // cudaMemcpy(buf.dev_pointer, buf.shm_buf, buf.size, cudaMemcpyHostToDevice);
    //cuMemsetD32(buf.dev_pointer, val, size);
    cuMemsetD8(buf.dev_pointer, val, size);
    CHECK_CUDA_ERROR();
    return 0;
}


int SlateClient::do_malloc()
{
    struct client_mem buf;
    CUdeviceptr addr; 
    int size;
    int buf_id;
    key_t key;
    key = std::hash<int>{}(std::stoi(pid_fifo) + rand());
    read(fdin, &size, sizeof(size_t));
    std::cout << "malloc_end: " << mytimestamp() << std::endl;
    CUDA_SAFE_CALL(cuMemAlloc(&addr, size));
    std::cout << "malloc_start: " << mytimestamp() << std::endl;
    //cuMemAlloc(&addr, size);
    buf_id = shmget(key, size, IPC_CREAT | IPC_EXCL | S_IRUSR | S_IWUSR | S_IWOTH | S_IROTH); 
    buf.shm_id = buf_id;
    buf.shm_buf = shmat(buf_id, 0, 0);
    buf.dev_pointer = addr;
    memset(buf.shm_buf, 0, size);
    buf.size = size;
    bufs.push_back(buf);
    keymap.insert(std::make_pair<>(key, buf));
    write(fdout, &key, sizeof(key_t));
    //printf("addr key: %lu\n", buf.dev_pointer);
    //printf("malloc key: %u\n", key);
    //printf("size: %d\n", size);
//cudaPointerAttributes atts1;
//cudaPointerGetAttributes ( &atts1, buf.dev_pointer);
CHECK_CUDA_ERROR();
//printf("arg memtype: %u\n", (const void*) atts1.memoryType);
    return 0;
}

#define H1(s,i,x)   (x*65599u+(uint8_t)s[(i)<strlen(s)?strlen(s)-1-(i):strlen(s)])
#define H4(s,i,x)   H1(s,i,H1(s,i+1,H1(s,i+2,H1(s,i+3,x))))
#define H16(s,i,x)  H4(s,i,H4(s,i+4,H4(s,i+8,H4(s,i+12,x))))
#define H64(s,i,x)  H16(s,i,H16(s,i+16,H16(s,i+32,H16(s,i+48,x))))
#define H256(s,i,x) H64(s,i,H64(s,i+64,H64(s,i+128,H64(s,i+192,x))))
#define HASH(s)    ((uint32_t)(H256(s,0,0)^(H256(s,0,0)>>16)))
#define BLACKSCHOLES HASH("/home/tnallen/build/NVIDIA_CUDA-8.0_Samples/4_Finance/BlackScholes2/BlackScholes_kernel.cuh")
#define RNG HASH("/home/tnallen/build/NVIDIA_CUDA-8.0_Samples/4_Finance/quasirandomGenerator2/quasirandomGenerator_kernel.cu")
#define TRANSPOSE HASH("/home/tnallen/build/NVIDIA_CUDA-8.0_Samples/6_Advanced/transpose2/transpose.cu")
#define GAUSSIAN HASH("/home/tnallen/build/gpu-rodinia/cuda/gaussian2/gaussian_kernel.cu")
#define MMUL HASH("/home/tnallen/build/NVIDIA_CUDA-8.0_Samples/0_Simple/matrixMul2/matrixMul.h")
#define BS_VAL    0b00001
#define RNG_VAL   0b00010
#define TS_VAL    0b00100
#define GAUS_VAL  0b01000
#define MMUL_VAL  0b10000

// 1 means can-corun-with... should be symmetric between two applications
#define BS_MASK    0b01011
#define RNG_MASK   0b11111
#define TS_MASK    0b00010
//#define TS_MASK    0b10110
#define GAUS_MASK  0b00011
#define MMUL_MASK  0b00010
//#define MMUL_MASK  0b00110
//#define MMUL_MASK  0b00110

/*
#define BS_MASK    0b11111
#define RNG_MASK   0b11111
#define TS_MASK    0b11111
#define GAUS_MASK  0b11111
#define MMUL_MASK  0b11111
*/
/*
#define BS_MASK    0b00000
#define RNG_MASK   0b00000
#define TS_MASK    0b00000
#define GAUS_MASK  0b00000
#define MMUL_MASK  0b00000
*/
// i think assumes only 2 applications
int get_mask(int bits)
{
    int mask = 0;
    if(bits & BS_VAL)   mask ^= BS_MASK;
    if(bits & RNG_VAL)  mask ^= RNG_MASK;
    if(bits & TS_VAL)   mask ^= TS_MASK;
    if(bits & GAUS_VAL) mask ^= GAUS_MASK;
    if(bits & MMUL_VAL) mask ^= MMUL_MASK;
    return mask;
}

bool SlateClient::canFit(char* codepath)
{
    bool fits = false;
    switch (HASH(codepath))
    {
        case BLACKSCHOLES:
            my_bits = BS_VAL;
        break;
        case RNG:
            my_bits = RNG_VAL;
        break;
        case TRANSPOSE:
            my_bits = TS_VAL;
        break;
        case GAUSSIAN:
            my_bits = GAUS_VAL;
        break;
        case MMUL:
            my_bits = MMUL_VAL;
        break;
        default:
            printf("idk this app\n");
        break;
    }
    printf ("mybits %d\n", my_bits);
    printf ("current_bits %d\n", current_bits);
    
    std::unique_lock<std::mutex> lock(query_lock);
    fits = !current_bits || (get_mask(current_bits) & my_bits);
    printf ("fits %d\n", fits);
    if (fits || !held_clients)
    {
        if (fits) running_clients++;
        else
        {
            held_clients++;
            fits = true;
        }
        can_launch.notify_one();
        current_bits |= my_bits; 
        if (running_clients > 1) printf ("CORUN!\n");
    }
    
    return fits;
}

void SlateClient::waitKernelFits(char* codepath)
{
    if (!canRun)
    {
        //printf("launch lock asked\n"); 
        std::unique_lock<std::mutex> lock(launch_lock);
        //printf("launch lock recvd\n"); 
        while(!canFit(codepath))
        {
            can_launch.wait(lock);
        }
        canRun = true;
        printf("held_clients2: %d\n", held_clients);
        printf("running_clients2: %d\n", running_clients);
        FairScheduler::update_sched();
        printf("%s can run\n", codepath);
    }
}

int SlateClient::handle_kernel(size_t reps)
{
    char codepath[STR_BUF_SIZE];
    char kname[STR_BUF_SIZE];
    dim3 blocks;
    dim3 threads;
    size_t smem;
    fgets(codepath, STR_BUF_SIZE, fdin);
    fgets(kname, STR_BUF_SIZE, fdin);
    read(fdin, &blocks, sizeof(dim3));
    read(fdin, &threads, sizeof(dim3));
    read(fdin, &smem, sizeof(size_t));
    //printf( "Codepath: %s\n", codepath);
    //printf( "Kname: %s\n", kname);
    //printf( "Blocks: %d %d %d\n", blocks.x, blocks.y, blocks.z);
    //printf( "Threads: %d %d %d\n", threads.x, threads.y, threads.z);
    //printf( "smem: %u\n", smem);

    size_t arg_size;
    std::list<std::pair<void*, bool>> args;
    while (true)
    {
        bool freeable = false;
        void* c;
        read(fdin, &arg_size, sizeof(size_t));
        if (arg_size == 0)
        {
            key_t key;
            read(fdin, &key, sizeof(key_t));
            if (key != 0)
            {
                struct client_mem& buf = keymap.at(key);
                c = &buf.dev_pointer;
            }
            else break;
        }
        else
        {
            c = (void*)new char[arg_size];
            freeable = true;
            read(fdin, c, arg_size);
        }
        args.emplace_back(c, freeable);
    }
    void* real_args[args.size()];
    int i = 0;
    std::list<std::pair<void*, bool>>::iterator it;
    for (it = args.begin(); it != args.end(); it++, i++)
    {
        real_args[i] = (*it).first;
    }
    std::string map_key = std::string(kname);
    SlateKernel* kernel;
    int num_kernels = kernels.size();
    if(kernels.count(map_key))
    {
        kernel = kernels.at(map_key);
        //kernel->updateBlocksAndThreads(blocks, threads);
    }
    else
    {
        kernel = new SlateKernel(codepath, kname, blocks, threads, smem, stream);
        kernels.emplace(map_key, kernel);
    }

    waitKernelFits(codepath);
    if (kernels.size() > 1 && kernels.size() > num_kernels) 
    {
        FairScheduler::update_sched();
    }
    //printf("Launch: %s\n", codepath);
    kernel->launch(real_args, args.size(), reps);
    while(!args.empty())
    {
        if(args.front().second) delete[] (char*) (args.front().first);
        args.pop_front();
    }
    return 0;
}

void SlateClient::reschedule()
{
    for (auto it : kernels)
    {
        it.second->reschedule();
    }
}
