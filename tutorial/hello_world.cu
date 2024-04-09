#include <stdio.h>

__global__ void hello(){
    printf("Hello CUDA World !!\n");
}

int main() {
    hello<<< 2, 4 >>>();
    cudaDeviceSynchronize();
    return 0;
}
