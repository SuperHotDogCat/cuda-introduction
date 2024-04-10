#include <stdio.h>
#include <cuda_runtime.h>

__global__ void checkIndex2d(){
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
    if (argc < 5){
        terminate("Usage check_dimention2d grid_size, block_size");
    }

    //device set up
    int device_id = 0;
    cudaSetDevice(device_id);

    int grid_size_x = atoi(argv[1]);
    int grid_size_y = atoi(argv[2]);
    int block_size_x = atoi(argv[3]);
    int block_size_y = atoi(argv[4]);

    dim3 grid(grid_size_x, grid_size_y);
    dim3 block(block_size_x, block_size_y);

    checkIndex2d<<<grid, block>>>();
    cudaDeviceSynchronize();
    return 0;
}