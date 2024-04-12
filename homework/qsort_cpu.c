#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int compareValue(const void *value1, const void *value2){
    int cmpvalue1 = *(int *) value1;
    int cmpvalue2 = *(int *) value2;
    if (cmpvalue1 > cmpvalue2){
        return 1;
    } else if (cmpvalue1 < cmpvalue2) {
        return -1;
    } else {
        return 0;
    }
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
    struct timespec start_time, end_time;
    unsigned long size = atol(argv[1]);
    srand(42); // initialize
    int *arr = initRandomArray(size);
    clock_gettime(CLOCK_REALTIME, &start_time);
    qsort(arr, size, sizeof(int), compareValue);
    clock_gettime(CLOCK_REALTIME, &end_time);
    //printArray(arr, size);
    printf("elapsed time %f\n", calculateElapsedTime(start_time, end_time));
}