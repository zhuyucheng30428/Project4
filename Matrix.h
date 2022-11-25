#ifndef MATRIX_H
#define MATRIX_H

struct Matrix{
    float * data;
    int row;
    size_t col;
};
struct Matrix *createMatrix(size_t row,size_t col);
struct Matrix *createMatrix_withData(float* data,size_t row,size_t col);
void printMatrix(const struct Matrix * a);
struct Matrix *inputMatrix(size_t row,size_t col);
void deleteMatrix(struct Matrix ** p);
struct Matrix *copyMatrix(const struct Matrix * a);
struct Matrix *addMatrix_withMatrix(const struct Matrix * a,const struct Matrix * b);
struct Matrix *subtractMatrix(const struct Matrix * a,const struct Matrix * b);
struct Matrix *multiplyMatrix(const struct Matrix * a,float n);
struct Matrix *addMatrix_withNumber(const struct Matrix * a,float n);
struct Matrix *matmul_improved(const struct Matrix * a,const struct Matrix * b);
struct Matrix *matmul_plain(const struct Matrix * a,const struct Matrix * b);
float findMatrixMin(const struct Matrix * a);
float findMatrixMax(const struct Matrix * a);
#endif
