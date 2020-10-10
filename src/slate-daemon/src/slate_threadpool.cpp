#include "slate_threadpool.h"

SlateThreadPool::SlateThreadPool(int num_threads, CUcontext ctx) : 
                                 done(false),
                                 size(num_threads),
                                 pool((std::thread*) malloc(sizeof(std::thread[num_threads]) * num_threads)),
                                 clients(ClientTable::getInstance()),
                                 context(ctx)
{
    for (int i = 0; i < size; i++)
    {
        new ((void*) &(pool[i])) std::thread(&SlateThreadPool::run_client, this);
    }
}

SlateThreadPool::~SlateThreadPool()
{
    done = true;
    cv.notify_all();
    if (pool)
    {
        for (int i = 0; i < size; i++)
        {
            pool[i].join();
        }
    }
    free(pool);
}

void SlateThreadPool::addClientTask(int pid)
{
    client_m.lock();
    task_pids.push_front(pid);
    client_m.unlock();
    cv.notify_one();
}


int SlateThreadPool::get_client_pid()
{
    int pid = 0;
    std::unique_lock<std::mutex> lock(client_m);
    cv.wait(lock, [&] { return !task_pids.empty() || done; });
    if (!task_pids.empty()) 
    {
        pid = task_pids.back();
        task_pids.pop_back();
    }
    return pid;
}

void SlateThreadPool::run_client()
{
    int pid;
    do
    {
        pid = get_client_pid();
        if (pid)
        {
            clients.lock_table();
            SlateClient& me = clients.add(pid, context);
            clients.unlock_table();
            int stat;
            while ((stat = me.process_cmds()) == 0);
            if (stat < 0)
            {
                fprintf(stderr, "Error for client %s\nErr: %d\n", me.getpid(), stat);
            }
            printf ("Good night %d\n", pid);
            clients.lock_table();
            clients.remove(me);
            clients.unlock_table();
        }
    }
    while (pid);
    printf("bye : (\n");
}
