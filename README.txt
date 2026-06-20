Project: Parallel Signal Processing 

Lazaros Evangelidis (2600057)

PREREQUISITES

To compile and run this project natively, your system needs:
A C++ compiler with OpenMP support (e.g., GCC/g++)
Intel ISPC compiler (Implicit SPMD Program Compiler)
A Linux environment (or WSL on Windows)

HOW TO COMPILE AND RUN

1)Compile the ISPC kernel (Vectorization / AVX2): 

ispc lpf.ispc -o lpf.o --target=avx2

2)Compile the C++ main file (Threading / OpenMP) and link the ISPC:

g++ main.cpp lpf.o -o main -O3 -fopenmp

3)Run the executable:

./main