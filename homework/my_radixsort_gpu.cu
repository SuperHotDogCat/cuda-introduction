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

__global__ void extractBitKernel(int *src_arr, int *cnt_arr, int b, long size){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx >= size) return;

    cnt_arr[idx + 1] = (src_arr[idx] >> b) & 1;
}

__global__ void prefixSumKernel(int *src_cnt_arr, int *dst_cnt_arr, int k, long size){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx >= size) return;

    dst_cnt_arr[idx + 1] = src_cnt_arr[idx + 1];
    if (idx - k + 1 >= 0) dst_cnt_arr[idx + 1] += src_cnt_arr[idx - k + 1];
}

__global__ void reorderKernel(int *src_arr, int *dst_arr, int *cnt_arr, long size){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx >= size) return;
    
    if (cnt_arr[idx] < cnt_arr[idx + 1])
        dst_arr[size - cnt_arr[size] + cnt_arr[idx]] = src_arr[idx];
    else
        dst_arr[idx - cnt_arr[idx]] = src_arr[idx];
}

void radixSort(int *arr, long size){
    int *src_arr, *dst_arr, *src_cnt_arr, *dst_cnt_arr;
    cudaMalloc((void**)&src_arr, sizeof(int) * size);
    cudaMalloc((void**)&dst_arr, sizeof(int) * size);
    cudaMalloc((void**)&src_cnt_arr, sizeof(int) * (size + 1));
    cudaMalloc((void**)&dst_cnt_arr, sizeof(int) * (size + 1));
    cudaMemcpy(src_arr, arr, sizeof(int) * size, cudaMemcpyHostToDevice);
    cudaMemset(src_cnt_arr, 0, sizeof(int));
    cudaMemset(dst_cnt_arr, 0, sizeof(int));

    int num_threads = 1024;

    dim3 block(num_threads);
    dim3 grid((size + block.x - 1) / block.x);

    for (int b = 0; b < sizeof(int) * 8 - 1; ++b) {
        extractBitKernel<<<grid, block>>>(src_arr, src_cnt_arr, b, size);
        cudaDeviceSynchronize();
        for (int k = 1; k < size; k <<= 1) {
            prefixSumKernel<<<grid, block>>>(src_cnt_arr, dst_cnt_arr, k, size);
            cudaDeviceSynchronize();
            
            // SWAP
            int* tmp = src_cnt_arr;
            src_cnt_arr = dst_cnt_arr;
            dst_cnt_arr = tmp;
        }
        reorderKernel<<<grid, block>>>(src_arr, dst_arr, src_cnt_arr, size);
        cudaDeviceSynchronize();

        // SWAP
        int* tmp = src_arr;
        src_arr = dst_arr;
        dst_arr = tmp;
    }
    cudaMemcpy(arr, src_arr, sizeof(int) * size, cudaMemcpyDeviceToHost);
}

int *initRandomArray(long size){
    int *arr = (int *)malloc(sizeof(int) * size);
    for (long i = 0; i < size; ++i){
        arr[i] = rand();
    }
    return arr;
}

void printArray(int *arr, long size){
    // debug 
    for (long i = 0; i < size; ++i){
        printf("%d ", arr[i]);
    }
    printf("\n");
}

double calculateElapsedTime(struct timespec start_time, struct timespec end_time){
    return (double) (end_time.tv_sec - start_time.tv_sec) + (double) (end_time.tv_nsec - start_time.tv_nsec) * 1e-9;
}

void terminate(const char *error_sentence){
    perror(error_sentence);
    exit(1);
}

int main(int argc, char **argv){
    if (argc < 2){
        terminate("Usage ./radixsort_gpu N");
    }
    //device set up
    int device_id = 0;
    cudaSetDevice(device_id);
    
    long size = atol(argv[1]);

    srand(42); // initialize
    int *arr = initRandomArray(size);

    struct timespec start_time, end_time;

    clock_gettime(CLOCK_REALTIME, &start_time);
    radixSort(arr, size);
    clock_gettime(CLOCK_REALTIME, &end_time);
    cudaCheckError();
    //printArray(arr, size);

    printf("elapsed time %f\n", calculateElapsedTime(start_time, end_time));
}