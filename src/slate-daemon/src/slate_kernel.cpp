#include "slate_kernel.h"
#include <chrono>
#include <algorithm>

/*
 #define INIT_IDX uint3 blockIdxx = blockIdx + bshift;\n\
 #define INIT_DIM dim3 gridDimm = gridDim + gshift;\n\
 #define blockIdx blockIdxx\n\
 #define gridDim gridDimm\n\
 #define GRIDDIM dim3 gridDimm\n\
 #define BLOCKIDX uint3 blockIdxx\n\
 #define GSHIFT dim3 gshift\n\
 #define BSHIFT uint3 bshift\n\
 ";
*/

typedef std::chrono::high_resolution_clock Clock;

SlateKernel::SlateKernel(const char* codepath, const char* kname, dim3 bs, dim3 ts, size_t s,
            cudaStream_t river) : blocks(bs), threads(ts), smem(s), 
                                  stream(river), kernel_name(kname)
{
        auto start = Clock::now();
        FairScheduler::get_params(sched_params);
        nvrtcProgram prog;
        compile(codepath, prog);
        linkandload(prog);
        auto end = Clock::now();
        std::cout << "compile_time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() << std::endl;
}

SlateKernel::~SlateKernel()
{
    cuModuleUnload(module);
}

void SlateKernel::updateBlocksAndThreads(dim3& bs, dim3& ts)
{
    blocks = bs;
    threads = ts;
    size_t bytes;// = sizeof(dim3);
    unsigned int total_blocks = bs.x * bs.y * bs.z;
    CUdeviceptr d_blocksx;
    CUdeviceptr d_blocksy;
    CUdeviceptr d_blocksz;
    CUdeviceptr d_threadsx;
    CUdeviceptr d_threadsy;
    CUdeviceptr d_threadsz;
    CUdeviceptr totalb;
    CUDA_SAFE_CALL(cuModuleGetGlobal(&d_blocksx, &bytes, module, "gdx"));
    CUDA_SAFE_CALL(cuModuleGetGlobal(&d_blocksy, &bytes, module, "gdy"));
    CUDA_SAFE_CALL(cuModuleGetGlobal(&d_blocksz, &bytes, module, "gdz"));
    CUDA_SAFE_CALL(cuModuleGetGlobal(&d_threadsx, &bytes, module, "bdx"));
    CUDA_SAFE_CALL(cuModuleGetGlobal(&d_threadsy, &bytes, module, "bdy"));
    CUDA_SAFE_CALL(cuModuleGetGlobal(&d_threadsz, &bytes, module, "bdz"));
    CUDA_SAFE_CALL(cuModuleGetGlobal(&totalb, &bytes, module, "total_blocks"));
    CUDA_SAFE_CALL(cuMemcpyHtoD(d_blocksx, &blocks.x, sizeof(unsigned int)));
    CUDA_SAFE_CALL(cuMemcpyHtoD(d_blocksy, &blocks.y, sizeof(unsigned int)));
    CUDA_SAFE_CALL(cuMemcpyHtoD(d_blocksz, &blocks.z, sizeof(unsigned int)));
    CUDA_SAFE_CALL(cuMemcpyHtoD(d_threadsx, &threads.x, sizeof(unsigned int)));
    CUDA_SAFE_CALL(cuMemcpyHtoD(d_threadsy, &threads.y, sizeof(unsigned int)));
    CUDA_SAFE_CALL(cuMemcpyHtoD(d_threadsz, &threads.z, sizeof(unsigned int)));
    CUDA_SAFE_CALL(cuMemcpyHtoD(totalb, &total_blocks, sizeof(unsigned int)));

}

void SlateKernel::reschedule()
{
    //printf ("Retreat!\n");
    CUdeviceptr retreat;
    size_t bytes;
    int val = 1;
    CUDA_SAFE_CALL(cuModuleGetGlobal(&retreat, &bytes, module, "retreat"));
    CHECK_CUDA_ERROR();
    cuMemcpyHtoDAsync(retreat, &val, sizeof(int), stream);
    CHECK_CUDA_ERROR();
}

void SlateKernel::launch(void* args[], size_t size, size_t reps)
{
        //CUdeviceptr gridDimm;
        //CUdeviceptr blockDimm;
        void* true_args[size + sched_params.size() + 1];
        //void* true_args[size + 1];
        //printf ("true_args size: %d\n", size + 1);
        CHECK_CUDA_ERROR();
        size_t i = 0;

        //CUDA_SAFE_CALL(cuModuleGetGlobal(&gridDimm, nullptr, module, "gridDimm"));
        //CUDA_SAFE_CALL(cuModuleGetGlobal(&blockDimm, nullptr, module, "blockDimm"));
        //CUDA_SAFE_CALL(cuMemcpyHtoD(gridDimm, &blocks, sizeof(dim3)));
        //CUDA_SAFE_CALL(cuMemcpyHtoD(blockDimm, &threads, sizeof(dim3)));
        true_args[0] = &reps;
        for (i = 1; i <= sched_params.size(); i++)
        {
            true_args[i] = &sched_params[i - 1];
        }
        for (i = 0; i < size; i++)
        {
            size_t adjusted_i = i + sched_params.size() + 1;
            true_args[adjusted_i] = args[i];
            //true_args[i] = args[i - 1];
        }
        //fprintf(stderr, "update_sched\n"); 
        //fprintf(stderr, "launch\n"); 
        CUDA_SAFE_CALL(cuLaunchKernel(kernel,
                                      1, 1, 1, // grid dim 
                                      1, 1, 1, // block dim 
                                      smem, stream, // shared mem and stream 
                                      true_args, NULL)); // arguments

}

void SlateKernel::compile(const char* codepath, nvrtcProgram& prog)
{
    const char* buffer;
    char* log;
    size_t logSize; 
    size_t sig_pos;
    size_t sig_end;
    const char *opts[] = {"-arch=compute_61", "-rdc=true", "--std=c++11"};//, "--maxrregcount=32"};
    //const char *opts[] = {"-arch=compute_61", "-rdc=true", "--std=c++11", "-G"};

    std::ifstream in(codepath);
    std::string contents((std::istreambuf_iterator<char>(in)), 
                          std::istreambuf_iterator<char>());



    //drop scheduler in
    sig_pos = contents.find(kernel_name);
    //printf("sigpos: %d\n", sig_pos);
    //printf("kernel_name: %s\n", kernel_name);
    sig_pos = contents.find("(", sig_pos + 1) + 1;
    sig_end = contents.find(")", sig_pos);
    std::cout << sig_pos << " " << sig_end << std::endl;
    std::string signature;
    if (sig_pos < sig_end)
    {
        signature = contents.substr(sig_pos, sig_end - sig_pos);
    }
    else
    {
        signature = std::string("");
    }
    //printf ("Signature: %s\n", signature.c_str());


    std::string sched = FairScheduler::gen_scheduler(signature, kernel_name, blocks, threads, smem);

    //for lex
    contents += "\0\0\0";
    buffer = contents.c_str();
    std::string final_kernel;
    //adding stuff
    void* lexer_state;
    init_scanner(buffer, &lexer_state);
    slate_yylex(final_kernel, lexer_state, blocks.y == 1);
    printf("Simple? blocks.y == %d\n", blocks.y);
    kill_scanner(lexer_state);

    //printf("threads: %d, %d, %d\n", threads.x, threads.y, threads.z);

    std::string inject("\ntypedef unsigned int uint;\n\
#define WARP_SZ 32\n\
__device__ inline int lane_id(void) { return threadIdx.x % WARP_SZ; }\n\
__device__ __inline__ uint get_smid(void)\n\
{\n\
    uint ret;\n\
    asm(\"mov.u32 %0, %smid;\" : \"=r\"(ret) );\n\
    return ret;\n\
}\n\
    __device__ volatile int retreat = 0;\n\
__device__ unsigned int slate_idx = 0;\n\
//__device__ volatile int living_blocks = 0;\n\
//__device__ const int slate_blocksize = " + std::to_string(threads.x * threads.y * threads.z) + ";\n\
__device__ const unsigned int slate_max = " + std::to_string(blocks.x * blocks.y) + ";\n\
__device__ const uint3 gridDimm = {" + std::to_string(blocks.x) + ", " + std::to_string(blocks.y) + ", " + std::to_string(blocks.z) +"};\n");

    final_kernel = inject + final_kernel + sched;
std::ofstream junk;
junk.open("sched.cu");
    junk << final_kernel << "\n";
junk.close();
    buffer = final_kernel.c_str();
    
/*   
    std::cout << "final_kernel" << std::endl; 
    std::cout << buffer << std::endl;
    std::cout << "codepath" << std::endl;
    std::cout << codepath << std::endl;
*/
    //printf("nvrtcCreateProgram\n");
    NVRTC_SAFE_CALL(nvrtcCreateProgram(&prog, // prog 
                       buffer, // buffer 
                       codepath, // name 
                       0,//1, // numHeaders 
                       0,//header, // headers 
                       NULL));//header_name); // includeNames
    //NVRTC_SAFE_CALL(nvrtcAddNameExpression(prog, kernel_name));
    nvrtcResult compileResult = nvrtcCompileProgram(prog, // prog 
                                                    3, // numOptions 

                                                    opts); // options
    nvrtcGetProgramLogSize(prog, &logSize); 
    log = new char[logSize]; 
    nvrtcGetProgramLog(prog, log); 
    std::cerr << "Logeroo: " << log << '\n';
    delete[] log;
    if (compileResult != NVRTC_SUCCESS) 
    {
        fprintf(stderr, "Compile failed.\n");
        throw nvrtc_exception();
    }
}

void SlateKernel::linkandload(nvrtcProgram& prog)
{
    CUlinkState linkState;
    char* ptx;
    void *cubin; 
    size_t ptxSize;
    size_t cubinSize; 
    CUjit_option_enum options[] = {CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES, CU_JIT_INFO_LOG_BUFFER, CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES};
    size_t err_size = 1000;
    size_t info_size = 1000;
    char err_log[err_size];
    char info_log[info_size];
    void* option_vals[] =  {(void*)err_log, (void*)&err_size, (void*)info_log, (void*)info_size};
    //printf("lowered name\n");
    //NVRTC_SAFE_CALL(nvrtcGetLoweredName(prog, kernel_name, &name));
    CUDA_SAFE_CALL(cuLinkCreate(0, options, option_vals, &linkState));
    //printf("cuLinkCreate\n");
    //CUDA_SAFE_CALL(cuLinkAddFile(linkState, CU_JIT_INPUT_OBJECT, "queue.o", 0, 0, 0));
    NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &ptxSize));
    //printf("PTXsize\n");
    ptx = new char[ptxSize];
    NVRTC_SAFE_CALL(nvrtcGetPTX(prog, ptx));
std::ofstream junk;
junk.open("sched.ptx");
    junk << ptx << "\n";
junk.close();
    
    //printf("PTX\n");
    NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));
    //fprintf(stderr, "1: %s\n", std::string(std::getenv("CUDA_PATH")).c_str()); 
    //fprintf(stderr, "2: %s\n", std::string("/lib64/libcudadevrt.a").c_str()); 
    std::string path = std::string(std::getenv("CUDA_PATH")) + std::string("/lib64/libcudadevrt.a");
    const char* env_p = path.c_str();
    //fprintf(stderr, "file: %s\n", env_p);
    CUDA_SAFE_CALL(cuLinkAddData(linkState, CU_JIT_INPUT_PTX, ptx, ptxSize, NULL, 0, 0, 0));
    CUDA_SAFE_CALL(cuLinkAddFile(linkState, CU_JIT_INPUT_LIBRARY, env_p, 0, 0, 0));
    //printf ("Error log:\n%s\n", err_log);
    //printf ("Info log:\n%s\n", info_log);
    //printf ("cubin size: %d\n", cubinSize);
    CUDA_SAFE_CALL(cuLinkComplete(linkState, &cubin, &cubinSize));
    //printf ("Error log:\n%s\n", err_log);
    //printf ("Info log:\n%s\n", info_log);
    CUDA_SAFE_CALL(cuModuleLoadData(&module, cubin));
    std::string kname = std::string(kernel_name) + std::string("scheduler");
    kname.erase( std::remove( kname.begin(), kname.end(), '<' ), kname.end() );
    kname.erase( std::remove( kname.begin(), kname.end(), '>' ), kname.end() );
    CUDA_SAFE_CALL(cuModuleGetFunction(&kernel, module, kname.c_str()));//"scheduler"));//kernel_name));
}
