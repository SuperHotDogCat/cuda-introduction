#include <stdio.h>
#include <stdlib.h>
#include <time.h> 

void initMatrix(double *mat, double init_num, int dim){
    // mat is expected to be a 2-dimentional matrix expressed by a 1-dimentional array. 
    // each dimention of mat is expected to be the same.
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j){
            mat[i*dim+j] = init_num;
        }
    }
}

void matMulCpu(double *input_mat1, double *input_mat2, double *output_mat, int dim){
    // mat is expected to be a 2-dimentional matrix expressed by a 1-dimentional array. 
    // each dimention of mat is expected to be the same.
    for (int i = 0; i < dim; ++i){
        for (int j = 0; j < dim; ++j){
            for (int k = 0; k < dim; ++k){
                output_mat[i*dim+j] += input_mat1[i*dim+k] * input_mat2[k*dim+j];
            }
        }
    }
}

double calculateElapsedTime(struct timespec start_time, struct timespec end_time){
    return (double) (end_time.tv_sec - start_time.tv_sec) + (double) (end_time.tv_nsec - start_time.tv_nsec) * 1e-9;
}

void terminate(const char *error_sentence){
    perror(error_sentence);
    exit(1);
}

void debugMatrix(double *mat, int dim){
    for (int i = 0; i < dim; ++i){
        for (int j = 0; j < dim; ++j){
            printf("%f ", mat[i*dim+j]);
        }
        printf("\n");
    }
}

int main(int argc, char **argv){
    if (argc < 2){
        terminate("Usage matmul_cpu dim_size");
    }

    int n = atoi(argv[1]);
    struct timespec start_time, end_time;
    double *input_mat1 = (double *)malloc(sizeof(double)*n*n);
    double *input_mat2 = (double *)malloc(sizeof(double)*n*n);
    double *output_mat = (double *)malloc(sizeof(double)*n*n);
    initMatrix(input_mat1, 3.0, n);
    initMatrix(input_mat2, 0.1, n);
    initMatrix(output_mat, 0.0, n);
    // start to measure time
    clock_gettime(CLOCK_REALTIME, &start_time);
    matMulCpu(input_mat1, input_mat2, output_mat, n);
    clock_gettime(CLOCK_REALTIME, &end_time);
    //debugMatrix(output_mat, n);
    printf("elapsed time %f\n", calculateElapsedTime(start_time, end_time));
}
