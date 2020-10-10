#pragma once
#include <thread>
#include <mutex>
#include <list>
#include <condition_variable>
#include <cuda.h>
#include "client.h"
#include "client_table.h"

class SlateThreadPool
{
    public:
        SlateThreadPool(int num_threads, CUcontext ctx);
        ~SlateThreadPool();
        SlateThreadPool()=delete;
        SlateThreadPool& operator=(SlateThreadPool&) = delete;

        void addClientTask(int pid);

    private:
        bool done;
        int size;
        std::thread* pool;
        std::mutex client_m;
        std::condition_variable cv;

        std::list<int> task_pids;
        ClientTable& clients;
        CUcontext context;

        int get_client_pid();

        void run_client();
};
