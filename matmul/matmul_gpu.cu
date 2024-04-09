void init_matrix(double *mat, double init_num, int dim){
    // mat is expected to be a 2-dimentional matrix expressed by a 1-dimentional array. 
    // each dimention of mat is expected to be the same.
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j){
            mat[i*dim+j] = init_num;
        }
    }
}

int main(int argc, char **argv){
    int i, j, n;
}