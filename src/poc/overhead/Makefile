all:
	nvcc -dc -arch=sm_35 test1.cu -o test1.o
	nvcc -arch=sm_35 test1.o -o test1
	nvcc -dc -arch=sm_35 test2.cu -o test2.o
	nvcc -arch=sm_35 test2.o -o test2

clean:
	rm *.o

	#nvcc -dc -arch=sm_35 -O0 -Xcicc -O0 -Xptxas -O0 test1.cu -o test1.o
	#nvcc -arch=sm_35 -O0 -Xcicc -O0 -Xptxas -O0 test1.o -o test1
	#nvcc -dc -arch=sm_35 -O0 -Xcicc -O0 -Xptxas -O0 test2.cu -o test2.o
	#nvcc -arch=sm_35 -O0 -Xcicc -O0 -Xptxas -O0 test2.o -o test2
