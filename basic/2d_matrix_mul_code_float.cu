#include <iostream>
#include <cuda_runtime.h>

__global__ void matrixMulKernel(const float* matrixA, const float* matrixB, float* matrixC, int rowsA, int colsA, int colsB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rowsA && col < colsB) {
        float sum = 0.0f;
        for (int k = 0; k < colsA; ++k) {
            sum += matrixA[row * colsA + k] * matrixB[k * colsB + col];
        }
        matrixC[row * colsB + col] = sum;

        // Print intermediate result
        printf("Intermediate result at [%d][%d]: %.2f\n", row, col, sum);

        // Print matrix A value
        printf("Matrix A value at [%d][%d]: %.2f\n", row, col, matrixA[row * colsA + col]);
    }
}

int main() {
    const int rowsA = 1;
    const int colsA = 784;
    const int colsB = 50;

    // Allocate memory for matrices on the host (CPU)
    float* hostMatrixA = new float[rowsA * colsA];
    float* hostMatrixB = new float[colsA * colsB];
    float* hostMatrixC = new float[rowsA * colsB];

    // Initialize matrices with sample values
    for (int i = 0; i < rowsA * colsA; ++i) {
        hostMatrixA[i] = 0.5f;
    }

    for (int i = 0; i < colsA * colsB; ++i) {
        hostMatrixB[i] = 0.5f;
    }

    // Allocate memory for matrices on the device (CUDA)
    float* deviceMatrixA;
    float* deviceMatrixB;
    float* deviceMatrixC;
    cudaMalloc(&deviceMatrixA, rowsA * colsA * sizeof(float));
    cudaMalloc(&deviceMatrixB, colsA * colsB * sizeof(float));
    cudaMalloc(&deviceMatrixC, rowsA * colsB * sizeof(float));

    // Copy the matrix data from host (CPU) to device (CUDA)
    cudaMemcpy(deviceMatrixA, hostMatrixA, rowsA * colsA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMatrixB, hostMatrixB, colsA * colsB * sizeof(float), cudaMemcpyHostToDevice);

    // Configure the CUDA kernel execution parameters
    dim3 blockSize(16, 16);
    dim3 gridSize((colsB + blockSize.x - 1) / blockSize.x, (rowsA + blockSize.y - 1) / blockSize.y);

    // Launch the CUDA kernel
    matrixMulKernel<<<gridSize, blockSize>>>(deviceMatrixA, deviceMatrixB, deviceMatrixC, rowsA, colsA, colsB);

    // Copy the result matrix from device (CUDA) to host (CPU)
    cudaMemcpy(hostMatrixC, deviceMatrixC, rowsA * colsB * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result matrix
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            std::cout << hostMatrixC[i * colsB + j] << " ";
        }
        std::cout << std::endl;
    }

    // Free the memory allocated on the host and device
    delete[] hostMatrixA;
    delete[] hostMatrixB;
    delete[] hostMatrixC;
    cudaFree(deviceMatrixA);
    cudaFree(deviceMatrixB);
    cudaFree(deviceMatrixC);

    return 0;
}
