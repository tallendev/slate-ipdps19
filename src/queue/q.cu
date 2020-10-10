#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <nvrtc.h>

//threads per block
#define TPB 256
#define SCHED_ERR

#define NVRTC_SAFE_CALL(x) \
        do { \
            nvrtcResult result = x; \
            if (result != NVRTC_SUCCESS) { \
                std::cerr << "\nerror: " #x " failed with error " \
                << nvrtcGetErrorString(result) << '\n'; \
                exit(1); \
            } \
         } while(0)

#define CUDA_SAFE_CALL(x) \
        do { \
            CUresult result = x; \
            if (result != CUDA_SUCCESS) { \
                const char *msg; \
                cuGetErrorName(result, &msg); \
                std::cerr << "\nerror: " #x " failed with error " \
                << msg << '\n'; \
                exit(1); \
            } \
        } while(0) 

#ifdef SCHED_ERR
#define CHECK_CUDA_ERROR()                                                    \
{                                                                             \
    cudaError_t err = cudaGetLastError();                                     \
    if (err != cudaSuccess)                                                   \
    {                                                                         \
        printf("error=%d name=%s at "                                         \
               "ln: %d\n  ",err,cudaGetErrorString(err),__LINE__);            \
        exit(-1);                                                        \
    }                                                                         \
}
#else
#define CHECK_CUDA_ERROR() 
#endif

void get_props(cudaDeviceProp* props)
{
    int count;
    cudaGetDeviceCount(&count);
    CHECK_CUDA_ERROR();
    if (!count)
    {
        fprintf(stderr, "No devices found. Bye!\n");
        exit(1);
    }
    cudaGetDeviceProperties(props, 0);
    CHECK_CUDA_ERROR();
}

void compile_and_run(int blocks)
{
    size_t ptxSize; 
    CUfunction kernel;
    CUmodule module;
    CUcontext ctx;
    CUdevice cuDevice; 
    CUlinkState linkState;

    cuDeviceGet(&cuDevice, 0);
    cuCtxCreate(&ctx, 0, cuDevice);


    nvrtcProgram prog;
    const char* buffer;
    const char *opts[] = {"-arch=compute_35", "-rdc=true"};
    std::ifstream in("test_code.cu");
    std::string contents((std::istreambuf_iterator<char>(in)), 
                          std::istreambuf_iterator<char>());
    contents.insert(0, "#include \"queue.h\"\n");
    //const char* header_name[] = { "queue.h" };
    //const char* header[] = { "__device__ void inc_val();\n__device__ int get_val();\n" };
    buffer = contents.c_str();
    NVRTC_SAFE_CALL(nvrtcCreateProgram(&prog, // prog 
                       buffer, // buffer 
                       "test_code.cu", // name 
                       0,//1, // numHeaders 
                       0,//header, // headers 
                       NULL));//header_name); // includeNames
    nvrtcResult compileResult = nvrtcCompileProgram(prog, // prog 
                                                    2, // numOptions 
                                                    opts); // options
    size_t logSize; 
    nvrtcGetProgramLogSize(prog, &logSize); 
    char *log = new char[logSize]; 
    nvrtcGetProgramLog(prog, log); 
    std::cerr << log << '\n';
    delete[] log;
    if (compileResult != NVRTC_SUCCESS) 
    {
        fprintf(stderr, "Compile failed. Bye!\n");
        exit(1); 
    }
    CUDA_SAFE_CALL(cuLinkCreate(0, 0, 0, &linkState));
    CUDA_SAFE_CALL(cuLinkAddFile(linkState, CU_JIT_INPUT_OBJECT, "queue.o", 0, 0, 0));

    NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &ptxSize));
    char *ptx = new char[ptxSize];
    NVRTC_SAFE_CALL(nvrtcGetPTX(prog, ptx));
    NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));

    CUDA_SAFE_CALL(cuLinkAddData(linkState, CU_JIT_INPUT_PTX, ptx, ptxSize, NULL, 0, 0, 0));
    size_t cubinSize; 
    void *cubin; 
    CUDA_SAFE_CALL(cuLinkComplete(linkState, &cubin, &cubinSize));
    CUDA_SAFE_CALL(cuModuleLoadData(&module, cubin));

    CUDA_SAFE_CALL(cuModuleGetFunction(&kernel, module, "test1"));
    
    int* devp;
    cudaMalloc(&devp, sizeof(int));
    void* args[] = { &devp };
    printf("Blocks: %d\nTPB:%d\n", blocks, TPB);
    cuLaunchKernel(kernel,
                   blocks, 1, 1, // grid dim 
                   TPB, 1, 1, // block dim 
                   0, NULL, // shared mem and stream 
                   args, 0); // arguments
    cudaDeviceSynchronize();
    int out = 0;
    cudaMemcpy(&out, devp, sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    printf("Out: %d\n", out);
}

int main()
{
    int blocks;
    cudaDeviceProp props;

    get_props(&props);
    blocks = (props.multiProcessorCount * props.maxThreadsPerMultiProcessor)/TPB; 
    
    compile_and_run(blocks);


    return 0;
}
