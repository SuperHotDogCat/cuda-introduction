#include <stdio.h>
// nvcc hello_world.cu -arch sm_86
__global__ void hello(){
    printf("Hello CUDA World !!\n");
}

int main() {
    hello<<< 2, 4 >>>();
    cudaDeviceSynchronize();
    return 0;
}
