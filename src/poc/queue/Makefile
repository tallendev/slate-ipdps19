all:
	nvcc -g -Xptxas -v -arch=sm_35 -dc q.cu -o q.o
	nvcc -g -Xptxas -v -arch=sm_35 -dc queue_code.cu -o queue.o
	nvcc -g -Xptxas -v -arch=sm_35 q.o queue.o -o q -lcuda -lnvrtc -lcudart

clean:
	rm -f *.o q
	
