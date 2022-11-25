#include <iostream>
extern"C"{
   #include "Matrix.h" 
}
#include <chrono>
using namespace std;

#define TIME_START start=std::chrono::steady_clock::now();
#define TIME_END(NAME) end=std::chrono::steady_clock::now(); \
             duration=std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();\
             cout<<(NAME)<<"duration = "<<duration<<"ms"<<endl;


int main(){
    size_t row,col;
    cout<<"Please input row and col"<<endl;
    cin>>row;
    cin>>col;
    row/=8;
    col/=8;
    struct Matrix *a= createMatrix(row,col);
    struct Matrix *b= createMatrix(row,col);
    //生成矩阵
    #pragma omp parallel num_threads(8)
    #pragma omp parallel for
    for(size_t i=0;i<row;i++){
        for(size_t j=0;j<col;j++){
            a->data[i*row+j]=i*row+j%100;
            b->data[i*row+j]=i*row+j%100;
        }
    }
    //计时
    auto start = std::chrono::steady_clock::now();
    auto end = std::chrono::steady_clock::now();
    auto duration = 0L;
    //矩阵乘法测试
    TIME_START;
    matmul_improved(a,b);
    TIME_END ("improved ")

    TIME_START;
    matmul_plain(a,b);
    TIME_END ("plain ")
    
    
    return 0;
}
