#include <stdio.h>
#include <cuda_runtime.h>

#define KS 3
#define IS 10


__global__
void convolutionKernel(const float *input, const float *kernel, float *output, int inputSize, int kernelSize) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < inputSize) {
        int halfKernelSize = kernelSize / 2;
        float result = 0.0f;

        for (int i = 0; i < kernelSize; ++i) {
            int inputIndex = tid - halfKernelSize + i;
            if (inputIndex >= 0 && inputIndex < inputSize) {
                result += input[inputIndex] * kernel[i];
            }
        }

        output[tid] = result;
    }
}

int main() {
    const int inputSize = IS;
    const int kernelSize = KS;

    float input[inputSize];
    float kernel[kernelSize] = {0.5, 1.0, 0.5};
    float output[inputSize];

    for (int i = 0; i < inputSize; i++)
        input[i] = i;

    printf("input: ");
    for (int i = 0; i < inputSize; i++)
        printf("%f ", input[i]);
    printf("\n");

    printf("kernel: ");
    for (int i = 0; i < kernelSize; i++)
        printf("%f ", kernel[i]);
    printf("\n");

    float *d_input, *d_kernel, *d_output;
    cudaMalloc((void**)&d_input, inputSize * sizeof(float));
    cudaMalloc((void**)&d_kernel, kernelSize * sizeof(float));
    cudaMalloc((void**)&d_output, inputSize * sizeof(float));

    cudaMemcpy(d_input, input, inputSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernelSize * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (inputSize + blockSize - 1) / blockSize;

    convolutionKernel<<<gridSize, blockSize>>>(d_input, d_kernel, d_output, inputSize, kernelSize);

    cudaMemcpy(output, d_output, inputSize * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);

    printf("output: ");
    for (int i = 0; i < inputSize; ++i) {
        printf("%f ", output[i]);
    }
    printf("\n");

    return 0;
}