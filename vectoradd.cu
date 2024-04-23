!nvcc --version

!pip install git+https://github.com/andreinechaev/nvcc4jupyter.git

%load_ext nvcc4jupyter

%%cuda
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

// CUDA kernel to add two vectors element-wise
__global__ void vectorAdd(const int *a, const int *b, int *c, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        c[tid] = a[tid] + b[tid];
    }
}

int main() {
    const int N = 1000; // Size of the vectors
    const int blockSize = 256; // Threads per block

    // Generate random vectors
    std::vector<int> hostA(N);
    std::vector<int> hostB(N);
    std::vector<int> hostC(N);

    srand(time(NULL));
    for (int i = 0; i < N; ++i) {
        hostA[i] = rand() % 1000; // Random numbers between 0 and 999
        hostB[i] = rand() % 1000;
    }

    // Allocate device memory
    int *deviceA, *deviceB, *deviceC;
    cudaMalloc((void**)&deviceA, N * sizeof(int));
    cudaMalloc((void**)&deviceB, N * sizeof(int));
    cudaMalloc((void**)&deviceC, N * sizeof(int));

    // Copy input vectors from host to device memory
    cudaMemcpy(deviceA, hostA.data(), N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB.data(), N * sizeof(int), cudaMemcpyHostToDevice);

    // Compute grid size
    int gridSize = (N + blockSize - 1) / blockSize;

    // Launch the kernel
    vectorAdd<<<gridSize, blockSize>>>(deviceA, deviceB, deviceC, N);

    // Copy result from device to host memory
    cudaMemcpy(hostC.data(), deviceC, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print original vectors and their sum vector
    std::cout << "Vector A: ";
    for (int i = 0; i < N; ++i) {
        std::cout << hostA[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Vector B: ";
    for (int i = 0; i < N; ++i) {
        std::cout << hostB[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Sum Vector: ";
    for (int i = 0; i < N; ++i) {
        std::cout << hostC[i] << " ";
    }
    std::cout << std::endl;

    // Free device memory
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

    return 0;
}
