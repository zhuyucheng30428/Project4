#ifndef MATRIX_H
#define MATRIX_H

typedef struct _Matrix{
    float * data;
    int row;
    size_t col;
} Matrix;
Matrix *createMatrix(size_t row,size_t col);
Matrix *createMatrix_withData(float* data,size_t row,size_t col);
Matrix *createMatrixRandom(const int row, const int col, const float leftBound, const float rightBound);
void printMatrix(const Matrix * a);
Matrix *inputMatrix(size_t row,size_t col);
void deleteMatrix(Matrix ** p);
Matrix *copyMatrix(const Matrix * a);
Matrix *addMatrix_withMatrix(const Matrix * a,const Matrix * b);
Matrix *subtractMatrix(const Matrix * a,const Matrix * b);
Matrix *multiplyMatrix(const Matrix * a,float n);
Matrix *addMatrix_withNumber(const Matrix * a,float n);
Matrix *matmul_improved(const Matrix * a,const Matrix * b);
Matrix *matmul_plain(const Matrix * a,const Matrix * b);
Matrix *matmul_openBLAS(const Matrix *mat1, const Matrix *mat2);
float findMatrixMin(const Matrix * a);
float findMatrixMax(const Matrix * a);
#endif
