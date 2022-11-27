// mac终端编译指令：
// echo 'export PATH="/opt/homebrew/opt/llvm/bin:$PATH"' >> ~/.bash_profile
// source ~/.bash_profile
// clang -O3 -o test matrix.c test.c -I /Users/tom/Documents/project4/openBLAS/include -L /Users/tom/Documents/project4/openBLAS/lib -lopenblas -fopenmp
#include <stdio.h>
# include "stdlib.h"
#include "Matrix.h"
# include "time.h"
# include "sys/time.h"
int main(int argc, char** argv) {
  srand(time(0));
  printf("total test number = %d\n", argc - 1);
  for (int i = 1; i < argc; i++) {
    struct timeval start,end; 
    long duration;
    int size = atoi(*(argv + i));
    printf("---------------------------------------------------\n");
    printf("test %d: size = %d\n", i, size);
    //生成矩阵
    Matrix *a = createMatrixRandom(size, size, -1e5, 1e5);
    Matrix *b = createMatrixRandom(size, size, -1e5, 1e5);
    Matrix *openblas, *improved,*plain;
    gettimeofday(&start, NULL);
    openblas = matmul_openBLAS(a, b);
    gettimeofday(&end, NULL);
    duration = 1000000*(end.tv_sec - start.tv_sec) + end.tv_usec-start.tv_usec;
    printf("time of openBLAS: %.3f ms\n", duration / 1000.0);

    gettimeofday(&start, NULL);
    improved = matmul_improved(a, b);
    gettimeofday(&end, NULL);
    duration = 1000000*(end.tv_sec - start.tv_sec) + end.tv_usec-start.tv_usec;
    printf("time of improved: %.3f ms\n", duration / 1000.0);

    gettimeofday(&start, NULL);
    plain = matmul_plain(a, b);
    gettimeofday(&end, NULL);
    duration = 1000000*(end.tv_sec - start.tv_sec) + end.tv_usec-start.tv_usec;
    printf("time of plain: %.3f ms\n", duration / 1000.0);
  }
  printf("---------------------------------------------------\n");
  return 0;
}
