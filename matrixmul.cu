!nvcc --version

!pip install git+https://github.com/andreinechaev/nvcc4jupyter.git

%load_ext nvcc4jupyter
%%cuda
#include <iostream>  // for input/output operations, 
#include <vector>    // for using vectors (arrays, lists etc)
#include <cuda_runtime.h> // provides CUDA runtime APIs for CUDA programming
using namespace std;

// Kernel function for matrix multiplication
__global__ void matrixMul(const int* A, const int* B, int* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        int sum = 0;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];  //IMPORTANT TO REMEMBER
        }
        C[row * N + col] = sum;  //IMPORTANT TO REMEMBER
    }
}

// Function to print a matrix
void printMatrix(const std::vector<int>& matrix, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << matrix[i * N + j] << " ";  //IMPORTANT TO REMEMBER
        }
        std::cout << std::endl;
    }
}

int main() {
    const int N = 3; // Size of the matrices (for demonstration purposes)
    const int blockSize = 2; // Threads per block
    const int gridSize = (N + blockSize - 1) / blockSize; // Number of blocks

    // Initialize host matrices
    std::vector<int> hostA = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<int> hostB = {9, 8, 7, 6, 5, 4, 3, 2, 1};
    std::vector<int> hostC(N * N);

    // Print input matrices
    std::cout << "Matrix A:" << std::endl;
    printMatrix(hostA, N);
    std::cout << "Matrix B:" << std::endl;
    printMatrix(hostB, N);

    // Declare device pointers
    int *deviceA, *deviceB, *deviceC;

    // Allocate device memory
    cudaMalloc((void**)&deviceA, N * N * sizeof(int));
    cudaMalloc((void**)&deviceB, N * N * sizeof(int));
    cudaMalloc((void**)&deviceC, N * N * sizeof(int));

    // Copy host data to device
    cudaMemcpy(deviceA, hostA.data(), N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB.data(), N * N * sizeof(int), cudaMemcpyHostToDevice);

    //IMPORTANT TO REMEMBER
    // Launch kernel                                  
    dim3 threadsPerBlock(blockSize, blockSize);
    dim3 numBlocks(gridSize, gridSize);
    matrixMul<<<numBlocks, threadsPerBlock>>>(deviceA, deviceB, deviceC, N);

    // Copy result back to host
    cudaMemcpy(hostC.data(), deviceC, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print result matrix
    std::cout << "Matrix C (A + B):" << std::endl;
    printMatrix(hostC, N);

    // Free device memory
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

    return 0;
}