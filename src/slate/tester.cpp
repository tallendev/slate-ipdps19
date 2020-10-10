#include <iostream>
#include <unistd.h>
#include <stdlib.h>
#include <string>
#include <slate.h>

const char code[] = "/home/tnallen/dev/slate/src/slate/test_cuda.cu";
const char cname[] = "test";

int main()
{
    size_t threads = 32;
    uint3 blocks;
    blocks.x = 32;
    blocks.y = 32;
    size_t bufsize = 500;

    char* mem;
    // cumalloc
    slateMalloc((void**)&mem, bufsize);
    strcpy(mem, "trombone\n");

    //cpytodevice
    slateCpyHtoD(mem);

    /// cu kernel
    slateLaunchKernel(code, cname, blocks, threads, 0, mem);
    //write(fdout, args, sizeof(args));

    slateSync();

    slateHangup();
    fprintf(stderr, "Tester done\n");
    return 0;
}
