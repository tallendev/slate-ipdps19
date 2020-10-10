#include <unistd.h>
#include <sys/wait.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>

int main(void)
{
    int pid = fork();
    if (pid == 0)
    {
        execl("/usr/local/cuda/bin/nvcc", "nvcc", "-ccbin", "/usr/local/gcc-5.4.0/bin/g++", "-arch=sm_61", "--std=c++11", "-ptx", "-rdc=true", "test.cu", "-o", "test.ptx", (char*)NULL);
        printf("execl errno: %d\n", errno);
        exit(0);
    }
    int idc;
    printf("waiting\n");
    wait(&idc);
    return 0;
}
