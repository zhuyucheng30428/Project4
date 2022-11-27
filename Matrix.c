#define WITH_NEON
#include <stdio.h>
#include <stdlib.h>
#include "Matrix.h"
#include "cblas.h"
#include </opt/homebrew/opt/libomp/include/omp.h>
#include "time.h"
#ifdef WITH_NEON
#include <arm_neon.h>
#endif



//创建空矩阵
Matrix *createMatrix(size_t row,size_t col){
    float * data = malloc(row*col * sizeof(float));
    Matrix * result=malloc (sizeof(Matrix));
    result->col=col;
    result->row=row;
    result->data=data;
    return result;
}
Matrix *createMatrix_withData(float* data,size_t row,size_t col){
    if(data){
        Matrix * result=malloc (sizeof(Matrix));
    result->col=col;
    result->row=row;
    result->data=data;
    return result;
    }
    else return createMatrix(row,col);
    
};
//打印矩阵
void printMatrix(const Matrix * a){
    if(a ){
        for(size_t i=0;i<a->row;i++){
            for(size_t j=0;j<a->col;j++){
                printf("%f ",a->data[i*a->col+j]);
            }
            printf("\n");
        }
    }
    else printf("Matrix to print doesn't exist!\n");
}
//输入矩阵
Matrix *inputMatrix(size_t row,size_t col){
    Matrix *result=createMatrix(row,col);
    printf("Please input a %zu by %zu matrix\n",row,col);
    for(size_t i=0;i<row*col;i++){
        scanf("%f",&result->data[i]);
    }
    printf("The input matrix is:\n");
    printMatrix(result);
    return result;
};
//删除矩阵
void deleteMatrix(Matrix ** p){
    if ((*p)->data) free((*p)->data); 
    if (*p) free(*p);
    *p=NULL;
    printf("Matrix deleted\n");
};
//复制矩阵
Matrix *copyMatrix(const Matrix * a){
    if(a){
       Matrix *result=createMatrix(a->row,a->col);
        for (size_t i=0;i<(a->row*a->col);i++){
            result->data[i]=a->data[i];
        }
        return result; 
    }
    else{
        printf("Matrix to copy doesn't exist!\n");
        return NULL;
    }
}
//生成随机矩阵
Matrix *createMatrixRandom(const int row, const int col, const float leftBound, const float rightBound) {
  Matrix *new_matrix = createMatrix(row, col);
  for (int i = 0; i < row; i++)
    for (int j = 0; j < col; j++) {
        new_matrix->data[(size_t) i * col + j] =
        leftBound + (rightBound - leftBound) * (float) rand() / (RAND_MAX + 1.0);
    }
  return new_matrix;
}
//矩阵加矩阵
Matrix *addMatrix_withMatrix(const Matrix * a,const Matrix * b){
    if (a && b){
       Matrix *result=copyMatrix(a);
        if(a->col==b->col &&a->row==b->row){
            for(size_t i=0;i<(a->row*a->col);i++){
                result->data[i]+=b->data[i];
            }
        }
        return result; 
    }
    else{
        printf("Matrix to add doesn't exist!\n");
        return NULL;
    }
};
//矩阵减矩阵
Matrix *subtractMatrix(const Matrix * a,const Matrix * b){
    if (a && b){
       Matrix *result=copyMatrix(a);
        if(a->col==b->col &&a->row==b->row){
            for(size_t i=0;i<(a->row*a->col);i++){
                result->data[i]-=b->data[i];
            }
        }
        return result; 
    }
    else{
        printf("Matrix to subtract doesn't exist!\n");
        return NULL;
    }
};
//矩阵乘以一个数
Matrix *multiplyMatrix(const Matrix * a,float n){
    if(a){
       Matrix *result=copyMatrix(a);
        for(size_t i=0;i<(a->row*a->col);i++){
            result->data[i]*=n;
        }
        return result; 
    }
    else {
        printf("Matrix to multiply doesn't exist!\n");
        return NULL;
    }
};
//矩阵加上一个数
Matrix *addMatrix_withNumber(const Matrix * a,float n){
    if(a){
       Matrix *result=copyMatrix(a);
        for(size_t i=0;i<(a->row*a->col);i++){
            result->data[i]+=n;
        }
        return result; 
    }
    else {
        printf("Matrix to add doesn't exist!\n");
        return NULL;
    }
};
//矩阵乘矩阵（优化）
Matrix *matmul_improved(const Matrix * mat1,const Matrix * mat2){
    #ifdef WITH_NEON
    if(mat1 && mat2){
        size_t row=mat1->row;
        size_t col=mat2->col;
        size_t common=mat1->col;
        if(mat1->col==mat2->row){
            Matrix *result=createMatrix(row,col);
            size_t common_mod4=common-(common%4);
            #pragma omp parallel for
            for(size_t m=0;m<row;m++){
                //common是a的列数，也是b的行数
                float * p1=&(mat1->data[m * common]);
                float * p2=&(mat2->data[0]);
                float * p3=&(result->data[m*col]);
                for(size_t n=0;n<col;n++){
                    float32x4_t a, b;
                    float32x4_t c = vdupq_n_f32(0);
                    float sum[4] = {0};
                    //是4倍数的部分
                    for (size_t i=0;i<common_mod4;i+=4){
                        a = vld1q_f32(p1 + i);
                        b = vld1q_f32(p2 + i*col);
                        c = vaddq_f32(c, vmulq_f32(a, b));
                    }
                    vst1q_f32(sum, c);
                    *p3=(sum[0]+sum[1]+sum[2]+sum[3]);
                    //除4剩余的部分
                    for (size_t i=common_mod4;i<common;i++){
                        *p3+=mat1->data[m*common+i]*mat2->data[i*col+n];
                    }
                    p3++;
                    p2++;
                }
            }
            return result;
        }
        else{
            printf("Size of A and B doesn't match!\n");
            return NULL;
        }
    }
    else {
        printf("Matrix to multiply doesn't exist!\n");
        return NULL;
    }
    #endif
    return NULL;
};

//矩阵乘矩阵（普通）
Matrix *matmul_plain(const Matrix * a,const Matrix * b){
    if(a && b){
        size_t row=a->row;
        size_t col=b->col;
        size_t common=a->col;
        if(a->col==b->row){
            Matrix *result=createMatrix(row,col);
            for(size_t m=0;m<row;m++){
                for(size_t n=0;n<col;n++){
                    float cache=0;
                    for (size_t i=0;i<common;i++){
                        cache+=a->data[m*common+i]*b->data[i*col+n];
                    }
                    result->data[m*col+n]=cache;
                }
            }
            return result;
        }
        else{
            printf("Size of A and B doesn't match!\n");
            return NULL;
        }
    }
    else {
        printf("Matrix to multiply doesn't exist!\n");
        return NULL;
    }
};

//矩阵乘矩阵（openblas)
Matrix *matmul_openBLAS(const Matrix *mat1, const Matrix *mat2) {
    if (mat1->col != mat2->row) {
        printf("Size of A and B doesn't match!\n");
        return NULL;
    }
    Matrix *new_matrix = createMatrix(mat1->row, mat2->col);
    cblas_sgemm(CblasRowMajor,CblasConjNoTrans, CblasNoTrans, mat1->row, mat2->col, mat1->col, 1, mat1->data, mat1->col, mat2->data, mat2->col, 0, new_matrix->data, mat2->col);
    return new_matrix;
}
//找最小值
float minValue(float* data,size_t len){
    if (data){
        float m=data[0];
        for(size_t i=0;i<len;i++){
            if (data[i]<m){
                m=data[i];
            }
        }
        return m;
    }
    else {
        printf("Data source doesn't exist!\n");
        return 0;
    }
}
float findMatrixMin(const Matrix * a){
    if (a){
       return minValue(a->data,a->col*a->row); 
    }
    else {
        printf("Matrix to search in doesn't exist!\n");
        return 0;
    }
}
//找最大值
float maxValue(float* data,size_t len){
    if (data){
        float m=data[0];
        for(size_t i=0;i<len;i++){
            if (data[i]>m){
                m=data[i];
            }
        }
        return m;
    }
    else {
        printf("Data source doesn't exist!\n");
        return 0;
    }
}
float findMatrixMax(const Matrix * a){
    if (a){
       return maxValue(a->data,a->col*a->row); 
    }
    else {
        printf("Matrix to search in doesn't exist!\n");
        return 0;
    }
}