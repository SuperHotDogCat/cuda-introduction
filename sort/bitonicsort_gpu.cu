#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h> 
#include <cuda_runtime.h>

//Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError() {                                          \
 cudaError_t e = cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(0); \
 }                                                                 \
}

__global__ void bitonicSortKernel(int *arr, int big_block, int mini_block, int d, long size){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx >= size || idx + d >= size) return;

    int up = ((idx) >> big_block & 2) == 0;
    int flag1 = (idx & d) == 0;
    int flag2 = (arr[idx] > arr[idx + d]) == up;
    if (flag1 && flag2){
        // SWAP
        int tmp = arr[idx + d];
        arr[idx + d] = arr[idx];
        arr[idx] = tmp;
    }
}

void bitonicSort(int *arr, long size, int log2_size){
    int *d_arr;
    cudaMalloc((void**)&d_arr, sizeof(int) * size);
    cudaMemcpy(d_arr, arr, sizeof(int) * size, cudaMemcpyHostToDevice);
    int num_threads = 32;
    assert(num_threads * num_threads < 1024 + 1);

    dim3 block(num_threads);
    dim3 grid((size+block.x-1)/block.x);

    for (int big_block = 0; big_block < log2_size; ++big_block){
        for (int mini_block = 0; mini_block <= big_block; ++mini_block){
            int d = 1 << (big_block - mini_block);
            bitonicSortKernel<<<grid, block>>>(d_arr, big_block, mini_block, d, size);
            cudaDeviceSynchronize();
        }
    }
    cudaCheckError();
    cudaMemcpy(arr, d_arr, sizeof(int) * size, cudaMemcpyDeviceToHost);
}

int isPow2(long N){
    // return 1 if N is power of 2 else 0
    assert(N >= 0);
    if (N == 0){
        return 0;
    }
    return (N & (N - 1)) == 0;
}

int log2N(long N){
    int log2_N = 0;
    while (N / 2 != 0){
        N = N / 2;
        log2_N += 1;
    }
    return log2_N;
}

int *initRandomArray(unsigned long size){
    int *arr = (int *)malloc(sizeof(int) * size);
    for (unsigned long i = 0; i < size; ++i){
        arr[i] = rand();
    }
    return arr;
}

double calculateElapsedTime(struct timespec start_time, struct timespec end_time){
    return (double) (end_time.tv_sec - start_time.tv_sec) + (double) (end_time.tv_nsec - start_time.tv_nsec) * 1e-9;
}

void printArray(int *arr, unsigned long size){
    // debug 
    for (unsigned long i = 0; i < size; ++i){
        printf("%d ", arr[i]);
    }
    printf("\n");
}

void terminate(const char *error_sentence){
    perror(error_sentence);
    exit(1);
}

int main(int argc, char **argv){
    if (argc < 2){
        terminate("Usage ./qsort_cpu N");
    }
    //device set up
    int device_id = 0;
    cudaSetDevice(device_id);
    
    unsigned long size = atol(argv[1]);
    assert(isPow2(size) == 1); // ensure size is power of 2.

    int log2_size = log2N(size);
    srand(42); // initialize
    int *arr = initRandomArray(size);

    struct timespec start_time, end_time;

    clock_gettime(CLOCK_REALTIME, &start_time);
    bitonicSort(arr, size, log2_size);
    clock_gettime(CLOCK_REALTIME, &end_time);
    //printArray(arr, size);

    printf("elapsed time %f\n", calculateElapsedTime(start_time, end_time));
}