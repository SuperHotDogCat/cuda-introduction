#include <stdio.h>
#include <stdlib.h>

void init_matrix(double *mat, double init_num, int dim){
    // mat is expected to be a 2-dimentional matrix expressed by a 1-dimentional array. 
    // each dimention of mat is expected to be the same.
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j){
            mat[i*dim+j] = init_num;
        }
    }
}

void matmul_cpu(double *input_mat1, double *input_mat2, double *output_mat, int dim){
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

int main(int argc, char **argv){
    int n;
    // add error processing here
    n = atoi(argv[1]);
    double *input_mat1 = (double *)malloc(sizeof(double)*n*n);
    double *input_mat2 = (double *)malloc(sizeof(double)*n*n);
    double *output_mat = (double *)malloc(sizeof(double)*n*n);
    init_matrix(input_mat1, 3.0, n);
    init_matrix(input_mat2, 0.1, n);
    init_matrix(output_mat, 0.0, n);
    // start to measure time
    matmul_cpu(input_mat1, input_mat2, output_mat, n);
    printf("ok\n");
}
