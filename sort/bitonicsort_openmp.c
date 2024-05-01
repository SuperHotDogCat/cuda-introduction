#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h> 
#include <omp.h>

// Fastest: OMP_NUM_THREADS=15 ./bitonicsort_openmp 16777216

void bitonicSort(int *arr, long size, int log2_size){
    for (int big_block = 0; big_block < log2_size; ++big_block){
        for (int mini_block = 0; mini_block <= big_block; ++mini_block){
            int d = 1 << (big_block - mini_block);
            #pragma omp parallel for
            for (int index = 0; index < size - d; ++index){
                int up = ((index) >> big_block & 2) == 0;
                int flag1 = (index & d) == 0;
                int flag2 = (arr[index] > arr[index+d]) == up;
                if (flag1 && flag2){
                    int tmp = arr[index+d];
                    arr[index+d] = arr[index];
                    arr[index] = tmp;
                }
            }
        }
    }
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
        terminate("Usage ./bitonicsort_openmp N");
    }
    
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
    free(arr);

    printf("elapsed time %f\n", calculateElapsedTime(start_time, end_time));
}