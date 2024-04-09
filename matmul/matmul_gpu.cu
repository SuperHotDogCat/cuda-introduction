#include <stdio.h>
#include <stdlib.h>
#include <time.h> 
#include <linux/time.h> // for my linux, #include<time.h>
#include <cuda_runtime.h>

/*

行列積の場合

for (int col = 0; i < nx; ++i){
    for (int row = 0; j < ny; ++j){
        // colとrowを用いた処理
    }
}

--------CUDA化--------

int col = threadIdx.y + blockDim.y * threadIdx.y;
int row = threadIdx.x + blockDim.x * threadIdx.x;
//colとrowを用いた処理

p59とかにあった
*/

void init_matrix(double *mat, double init_num, int dim){
    // mat is expected to be a 2-dimentional matrix expressed by a 1-dimentional array. 
    // each dimention of mat is expected to be the same.
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j){
            mat[i*dim+j] = init_num;
        }
    }
}


double calculate_elapsed_time(struct timespec start_time, struct timespec end_time){
    return (double) (end_time.tv_sec - start_time.tv_sec) + (double) (end_time.tv_nsec - start_time.tv_nsec) * 1e-9;
}

__device__ void matmul_gpu(double *input_mat1, double *input_mat2, double *output_mat, int dim){
    // mat is expected to be a 2-dimentional matrix expressed by a 1-dimentional array. 
    // each dimention of mat is expected to be the same.
    int i = threadIdx.y + blockDim.y * threadIdx.y;
    int j = threadIdx.x + blockDim.x * threadIdx.x;

    for (int k = 0; k < dim; ++k){
                output_mat[i*dim+j] += input_mat1[i*dim+k] * input_mat2[k*dim+j];
    }
}

int main(int argc, char **argv){
    int i, j, n;
}