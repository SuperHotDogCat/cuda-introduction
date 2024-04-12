#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <assert.h>
#include <time.h> 

#define NUM_DIGITS 10 // 桁数

int findMax(int *arr, long size){
    int max_number = INT_MIN;
    for (long i = 0; i < size; ++i){
        if (arr[i] > max_number){
            max_number = arr[i];
        }
    }

    return max_number;
}

void countingSort(int *arr, long size, int exp){
    int *output = (int*)calloc(size, sizeof(int));
    int *count = (int*)calloc(NUM_DIGITS, sizeof(int));

    for (long i = 0; i < size; ++i){
        int index = arr[i] / exp;
        count[index % NUM_DIGITS] += 1;
    }

    for (int i = 1; i < NUM_DIGITS; ++i){
        count[i] += count[i-1];
    }

    for (long i = size - 1; i >= 0; i--){
        int index = arr[i] / exp;
        output[count[index % 10] - 1] = arr[i];
        count[index % 10] -= 1;
    }

    for (long i = 0; i < size; ++i){
        arr[i] = output[i];
    }
}

void radixSort(int *arr,long size){
    int max_number = findMax(arr, size);

    int exp = 1;

    while (max_number / exp > 0){
        countingSort(arr, size, exp);
        exp *= 10;
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
        terminate("Usage ./radixsort N");
    }
    
    long size = atol(argv[1]);

    srand(42); // initialize
    int *arr = initRandomArray(size);

    struct timespec start_time, end_time;

    clock_gettime(CLOCK_REALTIME, &start_time);
    radixSort(arr, size);
    clock_gettime(CLOCK_REALTIME, &end_time);
    //printArray(arr, size);

    printf("elapsed time %f\n", calculateElapsedTime(start_time, end_time));
}