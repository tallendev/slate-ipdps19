#pragma once
class SlateClient;

#include <condition_variable>
#include <mutex>
#include <iostream>
#include <vector>
#include <list>
#include <functional>
#include <unordered_map>
#include <utility>
#include <mutex>
#include <stdio.h>
#include <string.h>
#include <sys/shm.h>                                                            
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#include "slate_comm.h"
#include "slate_kernel.h"
#include "safecuda.h"
#include "except.h"

// 256 bytes / 4? should mostly be pointers anyway...
#define MAX_ARGS 64
#define STR_BUF_SIZE 1024

#define fgets(path, size, fd)\
{\
    int i;\
    for (i = 0; i < size; i++) { read(fd, path+i, sizeof(char)); if(path[i] == '\0') break; }\
} while(0)

struct client_mem
{
    int shm_id;
    size_t size;
    void* shm_buf;
    CUdeviceptr dev_pointer;
};

class SlateClient
{
    public:
        // must must must be created on thread that will be used for calls
        SlateClient(int pid, CUcontext ctx);
        SlateClient(SlateClient&) = delete;
        //SlateClient& operator=(SlateClient&) = delete;

        bool operator==(const SlateClient& b);
        int process_cmds();
        int getCanRun()
        {
            return canRun;
        }
        void reschedule();
        const char* getpid() const;
        ~SlateClient();
        static int getRunningClients()
        {
            return running_clients;
        }
        static int getHeldClients()
        {
            return held_clients;
        }

        template <typename T>
        void update_param(size_t param, T t, T mod);

    private:
        std::string pid_fifo;
        std::string fifo_in;
        std::string fifo_out;
        int fdin;
        int fdout;
        std::list<client_mem> bufs;
        CUstream stream;
        CUcontext context;
        std::unordered_map<key_t, client_mem> keymap;
        std::unordered_map<std::string, SlateKernel*> kernels;
        
        bool canRun;
        int my_bits;
        volatile static int current_bits;
        volatile static int running_clients;
        volatile static int held_clients;
        static std::mutex launch_lock;
        static std::mutex query_lock;
        static std::condition_variable can_launch;

        void waitKernelFits(char*);
        void done_running();
        bool canFit(char*);
        int do_devcpy();
        int do_hostcpy();
        int do_memset();
        int do_malloc();

        int handle_kernel(size_t reps);
        SlateClient() = delete;
};


template <typename T>
void SlateClient::update_param(size_t param, T t, T mod __attribute__((unused)))
{   
    T val = t;
    for (auto it : kernels)
    {
        it.second->update_param(param, val);
    }
}
