#include <stdio.h>
#include <stdlib.h>
#include <Python.h>
#include <cuda_runtime.h>
#include <numpy/arrayobject.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION


struct Weights {
    float* matrix;
    int ndims;
    int *shape;
    long int size;
};

struct Inputs {
    float* matrix;
    int ndims;
    int *shape;
    long int size;
};


PyObject* read_numpy_file(const char* file_path) {
    // Import the Python module containing the function
    PyObject* numpy_module = PyImport_ImportModule("numpy");
    if (numpy_module == nullptr) {
        PyErr_Print();
        return nullptr;
    }

    // Get the reference to the function
    PyObject* numpy_function = PyObject_GetAttrString(numpy_module, "load");
    if (numpy_function == nullptr) {
        PyErr_Print();
        Py_DECREF(numpy_module);
        return nullptr;
    }

    // Create the arguments tuple
    PyObject* args = PyTuple_New(1);
    PyTuple_SetItem(args, 0, PyUnicode_FromString(file_path));

    // Call the Python function with the arguments
    PyObject* result = PyObject_CallObject(numpy_function, args);
    if (result == nullptr) {
        PyErr_Print();
        Py_DECREF(numpy_module);
        Py_DECREF(numpy_function);
        Py_DECREF(args);
        return nullptr;
    }

    // Print the shape of the NumPy array
    PyObject* shape = PyObject_GetAttrString(result, "shape");
    if (shape != nullptr) {
        PyObject* repr = PyObject_Repr(shape);
        const char* str = PyUnicode_AsUTF8(repr);
        printf("Shape: %s\n", str);
        Py_DECREF(repr);
        Py_DECREF(shape);
    } else {
        printf("Failed to get shape.\n");
    }

    // Clean up references
    Py_DECREF(numpy_module);
    Py_DECREF(numpy_function);
    Py_DECREF(args);
    return result;
}

PyArrayObject* read_weights_from_numpy(const char* file_path, int print) {
    // Call the Python function and get the PyObject reference to the NumPy array
    PyObject* numpy_array = read_numpy_file(file_path);
    if (numpy_array == nullptr) {
        // Handle error
        return nullptr;
    }

    // Use the PyObject reference to the NumPy array as needed
    // Example: Print the array
    if (print == 1){
        PyObject* repr = PyObject_Repr(numpy_array);
        const char* str = PyUnicode_AsUTF8(repr);
        printf("%s\n", str);
    }

    // Convert the result to a NumPy array
    PyArrayObject* array = reinterpret_cast<PyArrayObject*>(numpy_array);

    return array;
}

int get_numpy_ndims(PyArrayObject* array){
    // Get number of dimensions
    int ndim = PyArray_NDIM(array);
    printf("%d \n", ndim);
    return ndim;
}


long int get_numpy_size(PyArrayObject* array){
    // Get the total size of the array
    npy_intp total_size_intp = PyArray_SIZE(array);
    printf("%" NPY_INTP_FMT "\n", total_size_intp);
    
    long int total_size = static_cast<long int>(total_size_intp);
    return total_size;
}



float* convert_PyArrayObject_to_float(PyArrayObject* array, int print, int *shape, int ndim) {

    printf("values %d %d \n", PyArray_TYPE(array), NPY_DOUBLE);
    // Check the data type of the numpy array
    if (PyArray_TYPE(array) != NPY_FLOAT32) {
        printf("Input numpy array is not of type float.\n");
    }

    // Convert the weight into float
    float* matrix = static_cast<float*>(PyArray_DATA(array));

    // Printing weights for checking it after conversion 
    if (print == 1){
        if (ndim == 2){
            for (int i = 0; i < shape[0]; ++i) {
                if (i ==0){
                    for (int j = 0; j < shape[1]; ++j) {
                        // matrix[j+(i*shape[1])] = 0.5f;
                        if(j<3 || j>shape[1]-3){
                            printf("%d index %.5f ", j+(i*shape[1]), matrix[j+(i*shape[1])]);
                        }
                    }
                    printf("\n");
                }
            }
        }
        else if (ndim == 1){
            for (int i = 0; i < shape[0]; ++i) {
                // matrix[i] = 0.5f;
                if(i<5 || i>shape[0]-5){
                    printf("%d index %.5f ", i, matrix[i]);
                }
                
            }
            printf("\n");
        }
       
    }

    return matrix; 
}

float* move_weight_to_cuda(float* weights ,long int total_size){
    // Allocate CUDA device memory
    float* d_data;
    printf("total size %d \n", total_size);

    cudaError_t cudaStatus;
    cudaStatus = cudaMalloc((void**)&d_data, total_size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
        // Handle the error or return an error code
    }
    // Copy the array data from host (CPU) to device (CUDA)
    cudaStatus = cudaMemcpy(d_data, weights, total_size * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
        // Handle the error or return an error code
    }
    return d_data;
}

void get_numpy_shape(PyArrayObject* array, Weights& weights, int ndim){
    // Get the dimensions of the array
    npy_intp* shape = PyArray_DIMS(array);
    weights.shape = (int *)malloc(ndim * sizeof(int));
    for (int i = 0; i < ndim; i++) {
        weights.shape[i] = static_cast<int>(shape[i]);
    }
}

Weights read_weights(const char* file_path, int print){
    Weights weight;

    // reading numpy weights
    PyArrayObject* array = read_weights_from_numpy(file_path, print);
    if (array == nullptr) {
        // Handle error
        weight.matrix = nullptr;
        return weight;
    }

    printf("111.111\n");

    int ndims = get_numpy_ndims(array);
    get_numpy_shape(array, weight, ndims);

    printf("111.222\n");
    
    long int size = get_numpy_size(array);
    float* matrix = convert_PyArrayObject_to_float(array, print, weight.shape, ndims);
    
    printf("111.333\n");
    // Release the PyObject reference
    

    float* cuda_weights = move_weight_to_cuda(matrix, size);
    
    printf("111.444\n");

    printf("################################ \n");
    Py_DECREF(array);




    weight.ndims = ndims;
    weight.size = size;
    weight.matrix = cuda_weights;

    return weight;
}

void get_numpy_shape(PyArrayObject* array, Inputs& weights, int ndim){
    // Get the dimensions of the array
    npy_intp* shape = PyArray_DIMS(array);
    weights.shape = (int *)malloc(ndim * sizeof(int));
    for (int i = 0; i < ndim; i++) {
        weights.shape[i] = static_cast<int>(shape[i]);
    }
}

Inputs read_image(const char* file_path, int print){
    Inputs input;

    PyArrayObject* array = read_weights_from_numpy(file_path, print);
    if (array == nullptr) {
        // Handle error
        input.matrix = nullptr;
        return input;
    }

    int ndims = get_numpy_ndims(array);
    get_numpy_shape(array, input, ndims);
    
    long int size = get_numpy_size(array);
    float* images = convert_PyArrayObject_to_float(array, print, input.shape, ndims);
    
    // Release the PyObject reference
    float* cuda_images = move_weight_to_cuda(images, size);

    Py_DECREF(array);


    input.ndims = ndims;
    input.size = size;
    input.matrix = cuda_images;

    return input;
}

__global__ void matrixMulKernel(float* matrixA, float* matrixB, float* matrixC, int rowsA, int colsA, int colsB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rowsA && col < colsB) {
        float sum = 0.0f;
        for (int k = 0; k < colsA; k++) {
            sum += matrixA[row * colsA + k] * matrixB[k * colsB + col];
        }
        matrixC[row * colsB + col] = sum;

        // Print intermediate result
        printf("Intermediate result at rowsA %d, colsB %d  [%d][%d]: %.5f\n", rowsA, colsB, row, col, sum);
    }
}


int main() {
    // Initialize the Python interpreter
    Py_Initialize();

    // Ensure that NumPy is available
    import_array();
    printf("111\n");
    Weights fc1_w = read_weights("./mlp/mnist_mlp/model/fc1.weight.npy", 1);
    // Weights fc1_w = read_weights("./mlp/mnist_mlp/1.npy", 1);
    printf("222\n");
    Weights fc2_w = read_weights("./mlp/mnist_mlp/model/fc2.weight.npy", 0);
    Weights fc3_w = read_weights("./mlp/mnist_mlp/model/fc3.weight.npy", 0);
    printf("#############################\n");
    Weights fc1_b = read_weights("./mlp/mnist_mlp/model/fc1.bias.npy", 0);
    Weights fc2_b = read_weights("./mlp/mnist_mlp/model/fc2.bias.npy", 0);
    Weights fc3_b = read_weights("./mlp/mnist_mlp/model/fc3.bias.npy", 0);

    //Read image
    Inputs image = read_image("./mlp/mnist_mlp/images/1.npy", 0);
    // Inputs image = read_image("./mlp/mnist_mlp/2.npy", 0);

    // Finalize the Python interpreter
    Py_Finalize();

    float* C;
    float* matrixC;
    int rowsA = image.shape[0];
    int colsA = image.shape[1];
    int rowsB = fc1_w.shape[0];
    int colsB = fc1_w.shape[1];

    printf(" ROW A %d  COLS A %d \n",rowsA, colsA);
    printf(" ROW B %d  COLS B %d \n",rowsB, colsB);

    C = (float *)malloc(rowsA * colsB * sizeof(float));
    cudaMalloc((void **)&matrixC, rowsA * colsB * sizeof(float));

    dim3 blockSize(16, 16);
    dim3 gridSize((colsB + blockSize.x - 1) / blockSize.x, (rowsA + blockSize.y - 1) / blockSize.y);

    printf("image address %d \n",image.matrix);
    printf("fc1 address %d \n",fc1_w.matrix);
    matrixMulKernel<<<gridSize, blockSize>>>(image.matrix, fc1_w.matrix, matrixC, rowsA, colsA, colsB);
    
    cudaMemcpy(C, matrixC,  rowsA * colsB * sizeof(float), cudaMemcpyDeviceToHost);


    cudaDeviceSynchronize();

    printf("rowsA %d\n", rowsA);
    printf("colsB %d\n", colsB);
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++){
            // if (j < 5 || j > colsB-5){
                printf("%f ", C[i * rowsA + j]);
            // }
        }
        printf("\n");
    }


    sleep(5);
 
    // Clean up CUDA device memory
    cudaFree(fc1_w.matrix);
    cudaFree(fc2_w.matrix);
    cudaFree(fc3_w.matrix);
    
    cudaFree(fc1_b.matrix);
    cudaFree(fc2_b.matrix);
    cudaFree(fc3_b.matrix);

    return 0;
}