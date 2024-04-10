#include <stdio.h>
#include <cuda_runtime.h>

/*
Naming rule: function: Upper Camel, Variable: snake 
*/

__device__ void gpuAdd(int *number){
    *number += 1;
}

__global__ void callGpu(int *number){
    gpuAdd(number);
}

int main(){

    //device set up
    int device_id = 0;
    cudaSetDevice(device_id);
    //allocate memory on cpu
    int *g = (int*)malloc(sizeof(int));
    *g = 0;
    // allocate memory on gpu
    int *d_g = 0;
    cudaMalloc((void**) &d_g,sizeof(int));
    // memcpy host -> device
    cudaMemcpy(d_g, g, sizeof(int), cudaMemcpyHostToDevice); //これがUpperCaseじゃないのまじで納得行ってない
    // execute
    callGpu<<<1, 1>>>(d_g);
    cudaDeviceSynchronize(); // Wait until GPU processing finishs.
    cudaMemcpy(g, d_g, sizeof(int), cudaMemcpyDeviceToHost); 
    // free
    cudaFree(d_g);
    // display the answer
    printf("ans: %d \n", *g);
    return 0;
}