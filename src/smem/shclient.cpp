#include <stdlib.h>
#include <errno.h>
#include <unistd.h>
#include <stdio.h> 
#include <sys/shm.h> 
#include <sys/stat.h> 
 
int main () 
{
    int segment_id; 
    char* shared_memory; 
    struct shmid_ds shmbuffer; 
    int segment_size; 
    const int shared_segment_size = sizeof(char) * 128;
    const int skey = 0xC0FFEE;

    /* Allocate a shared memory segment.  */ 
    segment_id = shmget (skey, 0, 0666); 

    if (segment_id < 0)
    {
        printf("err seg id %d\n", errno);
        exit(1);
    }

    /* Attach the shared memory segment.  */ 
    shared_memory = (char*) shmat (segment_id, 0, 0); 
    printf ("shared memory attached at address %p\n", shared_memory); 
    /* Determine the segment's size. */ 
    shmctl (segment_id, IPC_STAT, &shmbuffer); 
    segment_size = shmbuffer.shm_segsz; 

    printf ("client segment size: %d\n", segment_size); 

    printf("Client: %s\n", shared_memory);
    
    /* Detach the shared memory segment.  */ 
    shmdt (shared_memory); 

    return 0; 
} 
