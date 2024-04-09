#include <stdio.h>
#include <cuda_runtime.h>

__global__ void checkIndex1d(){
    printf("threadIdx(%d, %d, %d) blockIdx(%d, %d, %d) blockDim(%d, %d, %d) gridDim(%d %d %d)\n", 
    threadIdx.x, threadIdx.y, threadIdx.z, 
    blockIdx.x, blockIdx.y, blockIdx.z, 
    blockDim.x, blockDim.y, blockDim.z,
    gridDim.x, gridDim.y, gridDim.z);
}

void terminate(const char *error_sentence){
    perror(error_sentence);
    exit(1);
}

int main(int argc, char **argv){
    if (argc < 3){
        terminate("Usage check_dimention1d grid_size, block_size");
    }

    //device set up
    int device_id = 0;
    cudaSetDevice(device_id);

    int grid_size = atoi(argv[1]);
    int block_size = atoi(argv[2]);

    dim3 grid(grid_size);
    dim3 block(block_size);

    checkIndex1d<<<grid, block>>>();
    cudaDeviceSynchronize();
    return 0;
}