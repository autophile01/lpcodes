!nvcc --version

!pip install git+https://github.com/andreinechaev/nvcc4jupyter.git

%load_ext nvcc4jupyter
%%cuda
#include <iostream>
#include <vector>
#include <cmath>

// CUDA kernel for parallel reduction to find the minimum value in an array
__global__ void minReduce(const int* arr, int* result, int N) {
    extern __shared__ int sharedData[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sharedData[tid] = (i < N) ? arr[i] : INT_MAX;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && i + s < N) {
            sharedData[tid] = min(sharedData[tid], sharedData[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        result[blockIdx.x] = sharedData[0];
    }
}

// CUDA kernel for parallel reduction to find the maximum value in an array
__global__ void maxReduce(const int* arr, int* result, int N) {
    extern __shared__ int sharedData[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sharedData[tid] = (i < N) ? arr[i] : INT_MIN;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && i + s < N) {
            sharedData[tid] = max(sharedData[tid], sharedData[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        result[blockIdx.x] = sharedData[0];
    }
}

// CUDA kernel for parallel reduction to find the sum of values in an array
__global__ void sumReduce(const int* arr, int* result, int N) {
    extern __shared__ int sharedData[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sharedData[tid] = (i < N) ? arr[i] : 0;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && i + s < N) {
            sharedData[tid] += sharedData[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        result[blockIdx.x] = sharedData[0];
    }
}

int main() {
    const int N = 512; // Number of elements in the array
    const int blockSize = 256; // Threads per block

    // Generate random array
    std::vector<int> hostArr(N);
    for (int i = 0; i < N; ++i) {
        hostArr[i] = rand() % 1000; // Generate random numbers between 0 to 999
    }

    // Allocate device memory
    int *deviceArr, *deviceResult;
    cudaMalloc((void**)&deviceArr, N * sizeof(int));
    cudaMalloc((void**)&deviceResult, sizeof(int));

    // Copy data from host to device
    cudaMemcpy(deviceArr, hostArr.data(), N * sizeof(int), cudaMemcpyHostToDevice);

    // Compute grid size
    int gridSize = (N + blockSize - 1) / blockSize;

    // Find minimum
    minReduce<<<gridSize, blockSize, blockSize * sizeof(int)>>>(deviceArr, deviceResult, N);
    int minVal;
    cudaMemcpy(&minVal, deviceResult, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "Minimum: " << minVal << std::endl;

    // Find maximum
    maxReduce<<<gridSize, blockSize, blockSize * sizeof(int)>>>(deviceArr, deviceResult, N);
    int maxVal;
    cudaMemcpy(&maxVal, deviceResult, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "Maximum: " << maxVal << std::endl;

    // Find sum
    sumReduce<<<gridSize, blockSize, blockSize * sizeof(int)>>>(deviceArr, deviceResult, N);
    int sumVal;
    cudaMemcpy(&sumVal, deviceResult, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "Sum: " << sumVal << std::endl;

    // Find average
    double average = static_cast<double>(sumVal) / N;
    std::cout << "Average: " << average << std::endl;

    // Free device memory
    cudaFree(deviceArr);
    cudaFree(deviceResult);

    return 0;
}