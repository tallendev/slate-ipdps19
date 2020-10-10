CC=g++ #clang++ #g++
LEX =flex
CFLAGS=-Wall -Wextra -std=c++11 -Wno-unused-function -march=native -O1
SOURCES=kernel_scan.yy.c client.cpp client_table.cpp slate_sched.cpp slate_kernel.cpp slate_threadpool.cpp slate-daemon.cpp
OBJ=client.o client_table.o slate_sched.o slate_kernel.o slate_threadpool.o kernel_scan.yy.o slate-daemon.o 
ID=../include
HEADER=$(ID)/client.h $(ID)/client_table.h $(ID)/except.h $(ID)/safecuda.h $(ID)/slate_comm.h $(ID)/slate_kernel.h $(ID)/slate_sched.h $(ID)/slate_threadpool.h
OBJDIR=../build
EXE=../slate-daemon
#INC=-I../include/ -I/home/tnalleninclude 
#LDFLAGS= -L/home/tnallen/lib -L/home/tnallen/lib64 -lcuda -lpthread -lcudart -lnvrtc
INC=-I../include/ -I/usr/local/cuda/include
LDFLAGS=-L/usr/local/cuda/lib64 -lcuda -lpthread -lcudart -lnvrtc
LEXFLAGS = -Wno-unused


all: $(OBJ) $(HEADER) $(SOURCES)
	$(CC) $(DEFS) $(CFLAGS) $(OBJ) -o $(EXE) $(LDFLAGS)

kernel_scan.yy.c: kernel_scan.l
	$(LEX) -o $@ $<

kernel_scan.yy.o: kernel_scan.yy.c
	$(CC) $(DEFS) $(INC) $(LEXFLAGS) -c $(CFLAGS) $< -o $@

.cpp.o: $(SOURCES) $(INC) $(HEADER)
	$(CC) $(DEFS) $(INC) -c $(CFLAGS) $< -o $@


clean:
	rm -f *.o kernel_scan.yy.c $(EXE)
