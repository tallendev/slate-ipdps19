#pragma once
#include <unordered_map>
#include <cstdarg>
#include <cstdlib>
#include <string>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/shm.h>                                                            
#include <sys/stat.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include "slate_comm.h"

#define mytimestamp() std::chrono::duration_cast< std::chrono::nanoseconds >(std::chrono::system_clock::now().time_since_epoch()).count()

#define fgets(path, size, fd)\
{\
    int i;\
    for (i = 0; i < size && path[i] != '\n'; i++) { read(fd, path+i, sizeof(char)); }\
    path[i] = '\0';\
}

std::unordered_map<void*, key_t> slate_bufs;
std::string inc_req_fifo = "/comm/def_fifo";
const int flags = 0666;
int fdout;
int fdin;
bool slate_init = 0;

static void slateInit()
{
    if (!slate_init)
    {
        std::cout << "init_start: " <<  mytimestamp() << std::endl;
        int fd;
       
        
        pid_t pid = getpid();
        std::string slate_dir = std::string(getenv("SLATE_DIR"));
        inc_req_fifo = slate_dir + inc_req_fifo;

        fd = open(inc_req_fifo.c_str(), O_WRONLY);
        write(fd, &pid, sizeof(pid_t));
        if (errno)
        {
            fprintf(stderr, "errno: %d\n", errno);
        }
        close(fd);
        std::string in_file = ("/comm/" + std::to_string(pid) + "toc");
        in_file = slate_dir + in_file;
        std::string out_file = ("/comm/" + std::to_string(pid) + "tos");
        out_file = slate_dir + out_file;
        const char* in = in_file.c_str();
        const char* out = out_file.c_str();
        //printf("out: %s\n", out);
        mkfifo(in, 0666);
        mkfifo(out, 0666);
        fdout = open(out, O_WRONLY);
        fdin = open(in, O_RDONLY);
        slate_init = 1;
        std::cout << "init_end: " <<  mytimestamp() << std::endl;
    }
}

void slateMalloc(void** pointer, size_t size)
{
    if (!slate_init) slateInit();
    std::cout << "malloc_start: " <<  mytimestamp() << std::endl;
    Command cmd;
    key_t key;
    int buf_id;

    cmd = Command::cumalloc;
    write(fdout, &cmd, sizeof(Command));
    write(fdout, &size, sizeof(size_t));

    read(fdin, &key, sizeof(key_t));
    buf_id = shmget(key, 0, flags);
    *pointer = shmat(buf_id, 0, 0);
    slate_bufs.insert(std::make_pair<>(*pointer, key));
    std::cout << "malloc_end: " <<  mytimestamp() << std::endl;
}


void slateMemset(void* pointer, int value, size_t size)
{
    std::cout << "memset_start: " <<  mytimestamp() << std::endl;
    Command cmd;
    key_t key = slate_bufs.at(pointer);
    cmd = Command::cumemset;
    write(fdout, &cmd, sizeof(Command));
    write(fdout, &key, sizeof(key_t));
    write(fdout, &value, sizeof(int));
    write(fdout, &size, sizeof(size_t));
    std::cout << "memset_end: " <<  mytimestamp() << std::endl;
}

void slateCpyHtoD(void* pointer)
{
    std::cout << "cpy2d_start: " <<  mytimestamp() << std::endl;
    Command cmd;
    key_t key = slate_bufs.at(pointer);
    cmd = Command::cpytodevice;
    write(fdout, &cmd, sizeof(Command));
    write(fdout, &key, sizeof(key_t));
    std::cout << "cpy2d_end: " <<  mytimestamp() << std::endl;
}

void slateCpyDtoH(void* pointer) 
{
    std::cout << "cpy2h_start: " <<  mytimestamp() << std::endl;
    Command cmd;
    key_t key = slate_bufs.at(pointer);
    cmd = Command::cpyfromdevice;
    SlateStatus s = SlateStatus::err;
    write(fdout, &cmd, sizeof(Command));
    write(fdout, &key, sizeof(key_t));
    read(fdin, &s, sizeof(SlateStatus));
    std::cout << "cpy2h_end: " <<  mytimestamp() << std::endl;
}

template <typename T>
static void do_args(T* t)
{
    size_t size;
    size = 0;
    write(fdout, &size, sizeof(size_t)); 
    void* p = t;
    key_t val = slate_bufs.at(p);
    write(fdout, &val, sizeof(key_t));
}

template <typename T>
static void do_args(T t)
{
    size_t size;
    size = sizeof(T);
    write(fdout, &size, sizeof(size_t)); 
    write(fdout, &t, sizeof(T));
}

template <typename T, typename... Args>
static void do_args(T t, Args... args)
{
    do_args(t);
    do_args(args...);
    
}

#define max_path 512
#define max_kern 128


template <typename... Args>
void slateLaunchKernel(const char* src_path, const char* kernel_name, size_t blocks, dim3 threads, size_t smem, Args... args)
{
    dim3 bs;
    bs.x = blocks;
    bs.y = 1;
    bs.z = 1;
    slateLaunchKernel(src_path, kernel_name, bs, threads, smem, args...); 
}


template <typename... Args>
void slateLaunchKernel(const char* src_path, const char* kernel_name, dim3 blocks, size_t threads, size_t smem, Args... args)
{
    dim3 ts;
    ts.x = threads;
    ts.y = 1;
    ts.z = 1;
    slateLaunchKernel(src_path, kernel_name, blocks, ts, smem, args...); 
}


template <typename... Args>
void slateLaunchKernel(const char* src_path, const char* kernel_name, size_t blocks, size_t threads, size_t smem, Args... args)
{
    dim3 bs;
    bs.x = blocks;
    bs.y = 1;
    bs.z = 1;
    dim3 ts;
    ts.x = threads;
    ts.y = 1;
    ts.z = 1;
    slateLaunchKernel(src_path, kernel_name, bs, ts, smem, args...); 
}


template <typename... Args>
void slateLaunchKernel(const char* src_path, const char* kernel_name, dim3 blocks, dim3 threads, size_t smem, Args... args) 
{
    slateLaunchBatchKernel(1, src_path, kernel_name, blocks, threads, smem, args...);
}

template <typename... Args>
void slateLaunchBatchKernel(size_t reps, const char* src_path, const char* kernel_name, size_t blocks, dim3 threads, size_t smem, Args... args)
{
    dim3 bs;
    bs.x = blocks;
    bs.y = 1;
    bs.z = 1;
    slateLaunchBatchKernel(reps, src_path, kernel_name, bs, threads, smem, args...); 
}

template <typename... Args>
void slateLaunchBatchKernel(size_t reps, const char* src_path, const char* kernel_name, dim3 blocks, size_t threads, size_t smem, Args... args)
{
    dim3 ts;
    ts.x = threads;
    ts.y = 1;
    ts.z = 1;
    slateLaunchBatchKernel(reps, src_path, kernel_name, blocks, ts, smem, args...); 
}


template <typename... Args>
void slateLaunchBatchKernel(size_t reps, const char* src_path, const char* kernel_name, size_t blocks, size_t threads, size_t smem, Args... args)
{
    dim3 bs;
    bs.x = blocks;
    bs.y = 1;
    bs.z = 1;
    dim3 ts;
    ts.x = threads;
    ts.y = 1;
    ts.z = 1;
    slateLaunchBatchKernel(reps, src_path, kernel_name, bs, ts, smem, args...); 
}


template <typename... Args>
void slateLaunchBatchKernel(size_t reps, const char* src_path, const char* kernel_name, dim3 blocks, dim3 threads, size_t smem, Args... args) 
{
    Command cmd;
    size_t zero = 0;
    cmd = Command::cukernel;
    if (!slate_init) slateInit();
    //fprintf(stderr, "%s, %s, %lu, %lu\n", src_path, kernel_name, blocks, threads);
    write(fdout, &cmd, sizeof(Command));
    write(fdout, &reps, sizeof(size_t));
    write(fdout, src_path, strnlen(src_path, max_path) + 1);
    write(fdout, kernel_name, strnlen(kernel_name, max_kern) + 1);
    write(fdout, &blocks, sizeof(dim3));
    write(fdout, &threads, sizeof(dim3));
    write(fdout, &smem, sizeof(size_t));
    do_args(args...);
    write(fdout, &zero, sizeof(size_t));
    write(fdout, &zero, sizeof(key_t));
}

void slateSync()
{
    Command cmd;
    SlateStatus s = SlateStatus::err;
    cmd = Command::cusync;

    if (!slate_init) slateInit();

    write(fdout, &cmd, sizeof(Command));
    read(fdin, (unsigned int*)&s, sizeof(unsigned int));
    std::cout << "sync_end: " <<  mytimestamp() << std::endl;
}

// later
void slateFree() 
{
    
}

void slateHangup()
{
    Command cmd;
    cmd = Command::hangup;
    write(fdout, &cmd, sizeof(Command));
}

#undef fgets
