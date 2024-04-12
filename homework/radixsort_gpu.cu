#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <assert.h>
#include <time.h>
#include <cuda_runtime.h>

#define NUM_DIGITS 10 // 桁数
#define BLOCK_SIZE 32 // ブロック内部のスレッド数

//Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError() {                                          \
 cudaError_t e = cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(0); \
 }                                                                 \
}

int findMax(int *arr, long size){
    int max_number = INT_MIN;
    for (long i = 0; i < size; ++i){
        if (arr[i] > max_number){
            max_number = arr[i];
        }
    }

    return max_number;
}

__global__ void updateCount(int *d_count, int *d_arr, int exp, long size){
    long i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= size) return;
    int index = d_arr[i] / exp;
    atomicAdd(&d_count[index % NUM_DIGITS], 1); // 強豪が起きていたので回避する
}

void countingSort(int *arr, long size, int exp){
    int *output = (int*)calloc(size, sizeof(int));
    int *count = (int*)calloc(NUM_DIGITS, sizeof(int));

    int *d_count;
    int *d_arr;
    cudaMalloc((void**)&d_count, NUM_DIGITS*sizeof(int));
    cudaMalloc((void**)&d_arr, size*sizeof(int));
    cudaMemcpy(d_count, count, NUM_DIGITS*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_arr, arr, size*sizeof(int), cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE);
    dim3 grid((size + block.x - 1) / block.x);

    updateCount<<<grid, block>>>(d_count, d_arr, exp, size);
    cudaDeviceSynchronize();

    cudaMemcpy(count, d_count, NUM_DIGITS*sizeof(int), cudaMemcpyDeviceToHost);


    // 本当に高速化したいのなら、以下のcumsumをさらに高速化したりしないといけない
    for (int i = 1; i < NUM_DIGITS; ++i){
        count[i] += count[i-1];
    }

    for (long i = size - 1; i >= 0; i--){
        int index = arr[i] / exp;
        output[count[index % NUM_DIGITS] - 1] = arr[i];
        count[index % NUM_DIGITS] -= 1;
    }
    

    memcpy(arr, output, size * sizeof(int));
}

void radixSort(int *arr,long size){
    int max_number = findMax(arr, size);

    int exp = 1;

    while (max_number / exp > 0){
        countingSort(arr, size, exp);
        exp *= NUM_DIGITS;

    }
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