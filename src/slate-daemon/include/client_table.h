#pragma once
#include <algorithm>
#include <mutex>
#include <list>
#include <cuda_runtime.h>
#include "client.h"

class ClientTable
{
    public:
        static ClientTable& getInstance();
        ~ClientTable(); 
        void remove(const SlateClient& it);
        size_t size();
        void lock_table();
        void unlock_table();

        template <typename T>
        void update_clients(size_t param, T t, T mod);
        void reschedule();

        SlateClient& add(int pid, CUcontext& context);
        ClientTable(ClientTable const&)    = delete;
        void operator=(ClientTable const&) = delete;

    private:
        std::list<SlateClient> clients;
        std::recursive_mutex cl_lock;
        
        ClientTable();
};

template <typename T>
void ClientTable::update_clients(size_t param, T t, T mod)
{
    std::for_each(clients.begin(), clients.end(), [&](SlateClient& c) 
    { 
        if (c.getCanRun())
        {
            c.update_param(param, t, mod); t+=mod;
        } 
    });
}
