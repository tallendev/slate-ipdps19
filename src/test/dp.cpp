#include <nvrtc.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>

#define NVRTC_SAFE_CALL(x)                                        \
do {                                                            \
nvrtcResult result = x;                                       \
if (result != NVRTC_SUCCESS) {                                \
std::cerr << "\nerror: " #x " failed with error "           \
<< nvrtcGetErrorString(result) << '\n';           \
exit(1);                                                    \
}                                                             \
} while(0)
#define CUDA_SAFE_CALL(x)                                         \
do {                                                            \
CUresult result = x;                                          \
if (result != CUDA_SUCCESS) {                                 \
const char *msg;                                            \
cuGetErrorName(result, &msg);                               \
std::cerr << "\nerror: " #x " failed with error "           \
<< msg << '\n';                                   \
exit(1);                                                    \
}                                                             \
} while(0)

const char *dynamic_parallelism = "                             \n\
extern \"C\" __global__                                         \n\
void child(float *out, size_t n)                                \n\
{                                                               \n\
size_t tid = blockIdx.x * blockDim.x + threadIdx.x;           \n\
if (tid < n) {                                                \n\
out[tid] = tid;                                             \n\
}                                                             \n\
}                                                               \n\
\n\
extern \"C\" __global__                                         \n\
void parent(float *out, size_t n,                               \n\
size_t numBlocks, size_t numThreads)              \n\
{                                                               \n\
child<<<numBlocks, numThreads>>>(out, n);                     \n\
cudaDeviceSynchronize();                                      \n\
}                                                               \n";
int main(int argc, char *argv[])
{
if (argc < 2) {
std::cout << "Usage: dynamic-parallelism <path to cudadevrt library>\n\n"
<< "<path to cudadevrt library> must include the cudadevrt\n"
<< "library name itself, e.g., Z:\\path\\to\\cudadevrt.lib on \n"
<< "Windows and /path/to/libcudadevrt.a on Linux and Mac OS X.\n";
exit(1);
}
size_t numBlocks = 32;
size_t numThreads = 128;
// Create an instance of nvrtcProgram with the code string.
nvrtcProgram prog;
NVRTC_SAFE_CALL(
nvrtcCreateProgram(&prog,                       // prog
dynamic_parallelism,         // buffer
"dynamic_parallelism.cu",    // name
0,                           // numHeaders
NULL,                        // headers
NULL));                      // includeNames
// Compile the program for compute_35 with rdc enabled.
const char *opts[] = {"--gpu-architecture=compute_35",
"--relocatable-device-code=true"};
nvrtcResult compileResult = nvrtcCompileProgram(prog,  // prog
2,     // numOptions
opts); // options
// Obtain compilation log from the program.
size_t logSize;
NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &logSize));
char *log = new char[logSize];
NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, log));
std::cout << log << '\n';
delete[] log;
if (compileResult != NVRTC_SUCCESS) {
exit(1);
}
// Obtain PTX from the program.
size_t ptxSize;
NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &ptxSize));
char *ptx = new char[ptxSize];
NVRTC_SAFE_CALL(nvrtcGetPTX(prog, ptx));
// Destroy the program.
NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));
// Load the generated PTX and get a handle to the parent kernel.
CUdevice cuDevice;
CUcontext context;
CUlinkState linkState;
CUmodule module;
CUfunction kernel;
CUDA_SAFE_CALL(cuInit(0));
CUDA_SAFE_CALL(cuDeviceGet(&cuDevice, 0));
CUDA_SAFE_CALL(cuCtxCreate(&context, 0, cuDevice));
CUDA_SAFE_CALL(cuLinkCreate(0, 0, 0, &linkState));
CUDA_SAFE_CALL(cuLinkAddFile(linkState, CU_JIT_INPUT_LIBRARY, argv[1],
0, 0, 0));
CUDA_SAFE_CALL(cuLinkAddData(linkState, CU_JIT_INPUT_PTX,
(void *)ptx, ptxSize, "dynamic_parallelism.ptx",
0, 0, 0));
size_t cubinSize;
void *cubin;
CUDA_SAFE_CALL(cuLinkComplete(linkState, &cubin, &cubinSize));
CUDA_SAFE_CALL(cuModuleLoadData(&module, cubin));
CUDA_SAFE_CALL(cuModuleGetFunction(&kernel, module, "parent"));
// Generate input for execution, and create output buffers.
size_t n = numBlocks * numThreads;
size_t bufferSize = n * sizeof(float);
float *hOut = new float[n];
CUdeviceptr dX, dY;
void* dOut;
cudaMalloc(&dOut, bufferSize);
// Execute parent kernel.
void *args[] = { dOut, &n, &numBlocks, &numThreads };
CUDA_SAFE_CALL(
cuLaunchKernel(kernel,
1, 1, 1,    // grid dim
1, 1, 1,    // block dim
0, NULL,    // shared mem and stream
args, 0));  // arguments
CUDA_SAFE_CALL(cuCtxSynchronize());
// Retrieve and print output.
//CUDA_SAFE_CALL(cuMemcpyDtoH(hOut, dOut, bufferSize));
/*
for (size_t i = 0; i < n; ++i) {
std::cout << hOut[i] << '\n';
}
// Release resources.
CUDA_SAFE_CALL(cuMemFree(dOut));
CUDA_SAFE_CALL(cuModuleUnload(module));
CUDA_SAFE_CALL(cuLinkDestroy(linkState));
CUDA_SAFE_CALL(cuCtxDestroy(context));

*/
delete[] hOut;
return 0;
}
