#include <unistd.h>
#include <stdio.h> 
#include <stdlib.h>
#include <sys/shm.h> 
#include <sys/stat.h> 
#include <errno.h>
 
int main () 
{
    int segment_id; 
    char* shared_memory; 
    struct shmid_ds shmbuffer; 
    int segment_size; 
    const int shared_segment_size = sizeof(char) * 128;
    const int skey = 0xC0FFEE;

    /* Allocate a shared memory segment.  */ 
    segment_id = shmget (skey, shared_segment_size, IPC_CREAT | IPC_EXCL | S_IRUSR | S_IWUSR); //| S_IWOTH | S_IROTH); 
    
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

    printf ("server segment size: %d\n", segment_size); 
    
    /* Write a string to the shared memory segment.  */ 
    sprintf (shared_memory, "Hello, world."); 

    printf("Server: %s\n", shared_memory);
    
    sleep(10);
    
    /* Detach the shared memory segment.  */ 
    shmdt (shared_memory); 

    /* Deallocate the shared memory segment.  */ 
    shmctl (segment_id, IPC_RMID, 0); 

    return 0; 
} 

