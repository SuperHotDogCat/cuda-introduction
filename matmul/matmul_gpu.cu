#include <stdio.h>
#include <stdlib.h>
#include <time.h> 
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

void initMatrix(double *mat, double init_num, int dim){
    // mat is expected to be a 2-dimentional matrix expressed by a 1-dimentional array. 
    // each dimention of mat is expected to be the same.
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j){
            mat[i*dim+j] = init_num;
        }
    }
}


double calculateElapsedTime(struct timespec start_time, struct timespec end_time){
    return (double) (end_time.tv_sec - start_time.tv_sec) + (double) (end_time.tv_nsec - start_time.tv_nsec) * 1e-9;
}

__global__ void matMulGpu(double *input_mat1, double *input_mat2, double *output_mat, int dim){
    // mat is expected to be a 2-dimentional matrix expressed by a 1-dimentional array. 
    // each dimention of mat is expected to be the same.
    int i = threadIdx.y + blockDim.y * threadIdx.y;
    int j = threadIdx.x + blockDim.x * threadIdx.x;

    if (i >= dim || j >= dim) return;
    // 3重ループ可能では?
    for (int k = 0; k < dim; ++k){
                output_mat[i*dim+j] += input_mat1[i*dim+k] * input_mat2[k*dim+j];
    }
}

void terminate(const char *error_sentence){
    perror(error_sentence);
    exit(1);
}

void debug_matrix(double *mat, int dim){
    for (int i = 0; i < dim; ++i){
        for (int j = 0; j < dim; ++j){
            printf("%f ", mat[i*dim+j]);
        }
        printf("\n");
    }
}

int main(int argc, char **argv){
    if (argc < 2){
        terminate("Usage check_dimention1d dim_size");
    }

    //device set up
    int device_id = 0;
    cudaSetDevice(device_id);

    int n = atoi(argv[1]);
    struct timespec start_time, end_time;
    double *input_mat1 = (double *)malloc(sizeof(double)*n*n);
    double *input_mat2 = (double *)malloc(sizeof(double)*n*n);
    double *output_mat = (double *)malloc(sizeof(double)*n*n);

    initMatrix(input_mat1, 3.0, n);
    initMatrix(input_mat2, 0.1, n);
    initMatrix(output_mat, 0.0, n);

    double *d_input_mat1, *d_input_mat2, *d_output_mat;
    cudaMalloc((void**) &d_input_mat1,sizeof(double)*n*n);
    cudaMalloc((void**) &d_input_mat2,sizeof(double)*n*n);
    cudaMalloc((void**) &d_output_mat,sizeof(double)*n*n);

    cudaMemcpy(d_input_mat1, input_mat1, sizeof(double)*n*n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_mat2, input_mat2, sizeof(double)*n*n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_mat, output_mat, sizeof(double)*n*n, cudaMemcpyHostToDevice);

    dim3 block(n, n);
    dim3 grid((n+block.x-1)/block.x, (n+block.y-1)/block.y);
    clock_gettime(CLOCK_REALTIME, &start_time);
    matMulGpu<<<grid, block>>>(d_input_mat1, d_input_mat2, d_output_mat, n);
    cudaMemcpy(output_mat, d_output_mat, sizeof(double)*n*n, cudaMemcpyDeviceToHost); 
    cudaDeviceSynchronize(); // Wait until GPU processing finishs.
    clock_gettime(CLOCK_REALTIME, &end_time);
    cudaFree(d_input_mat1);
    cudaFree(d_input_mat2);
    cudaFree(d_output_mat);

    debug_matrix(output_mat, n); // weird output
    return 0;
}