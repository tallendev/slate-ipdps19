#include <iostream>
#include <thread>

#include <execinfo.h>
#include <unistd.h>
#include <stdlib.h>
#include <fcntl.h>
#include <signal.h>
#include <future>

#include <sys/stat.h>

#include <cuda.h>
#include <nvrtc.h>

#include "slate_threadpool.h"
#include "safecuda.h"

#define POOL_SIZE 4


//#include "client.h"
//#include "client_table.h"

// fork off from the parent and gain eternal life
void fork_daemon()
{
/*
    pid_t pid;
    // Fork off the parent process     
    pid = fork();
    if (pid < 0)
    {
        exit(EXIT_FAILURE);
    }
    //If we got a good PID, then we can exit the parent process.
    if (pid > 0)
    {
        exit(EXIT_SUCCESS);
    }
*/
}

// fifo name tbd
const char* inc_req_fifo = "comm/def_fifo";
// fifo flags
const int flags = 0666;
// table of client references
//ClientTable& clients = ClientTable::getInstance();
// for safe keeping
CUcontext context;
CUdevice cuDevice; 

/*
void run_client(int pid)
{
    SlateClient& me = clients.add(pid, context);
    int stat;
    while ((stat = me.process_cmds()) == 0);
    if (stat < 0)
    {
        fprintf(stderr, "Error for client %s\nErr: %d\n", me.getpid(), stat);
    }
//    printf ("Good night %d\n", pid);
    clients.remove(me);
}
*/

// still going?
bool active = true;
int fd;

// clean up fifo :)
static void sig_handler(__attribute__((unused)) int sig)
{
    /*
    void *array[50];
    size_t size;

    // get void*'s for all entries on the stack
    size = backtrace(array, 50);
      
    // print out all the frames to stderr
    fprintf(stderr, "Error: signal %d:\n", sig);
    backtrace_symbols_fd(array, size, STDERR_FILENO);
    */
    printf("Sig caught\n");
    if (active)
    {
        printf("cleaning up\n");
        active = false;
        close(fd);
        unlink(inc_req_fifo);
    }
    printf("Sig done\n");
    exit(0);
}

// just grab all the signals, we just want to clean up the fifo before death
#define NUM_SIGNALS 30
struct sigaction sa;
static void set_sighandler()
{
    sa.sa_handler = sig_handler;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGTERM, &sa, NULL);
    sigaction(SIGINT, &sa, NULL);
}

std::future<void> sigsync;
// recieve new client requests.
/*
    for (int i = 1; i <= NUM_SIGNALS; i++)
    {
        sigaction(i, &sa, NULL);
    }
*/
int recv_clients()
{
    pid_t new_pid = 0;

    mkfifo(inc_req_fifo, flags);
    fd = open(inc_req_fifo, O_RDWR);
    
    SlateThreadPool pool(POOL_SIZE, context);
    
    while (active)
    {
        new_pid = 0;
        read(fd, &new_pid, sizeof(pid_t));
        if (!new_pid)
        {
            fprintf(stderr, "Error reading from pipe."
                            "\npid_recv: %d\n", new_pid);
            active = 0;
        }
        else
        {
            fprintf (stderr, "pid received: %d\n", new_pid);
            pool.addClientTask(new_pid);
            // size of list before addition is the id of this new thread
            //std::thread new_thread = std::thread(run_client, new_pid);
            //new_thread.detach();
        }
    }

    //fclose(fp);
    //unlink(inc_req_fifo);
    return 0;
}

void init_cuda()
{
    //CUDA_SAFE_CALL(cuInit(0));
    cudaSetDeviceFlags(0);
    CUDA_SAFE_CALL(cuDeviceGet(&cuDevice, 0)); 
    CUDA_SAFE_CALL(cuDevicePrimaryCtxRetain(&context, cuDevice));
    //CUDA_SAFE_CALL(cuCtxCreate(&context, 0, cuDevice));
}

int main()
{
    fork_daemon();
    sigsync = std::async(set_sighandler);
    srand(time(NULL));
    init_cuda();
    recv_clients();   
    
    return 0;
}
