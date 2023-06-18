#include <Python.h>
#include <numpy/arrayobject.h>

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
    if (print == 0){
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

npy_intp* get_numpy_shape(PyArrayObject* array){
    // Get the dimensions of the array
    npy_intp* shape = PyArray_DIMS(array);
    
    printf("%" NPY_INTP_FMT "\n", shape[0]);
    printf("%" NPY_INTP_FMT "\n", shape[1]);

    return shape;
}

long int get_numpy_size(PyArrayObject* array){
    // Get the total size of the array
    npy_intp total_size_intp = PyArray_SIZE(array);
    printf("%" NPY_INTP_FMT "\n", total_size_intp);
    
    long int total_size = static_cast<long int>(total_size_intp);
    return total_size;
}



float* convert_PyArrayObject_to_float(PyArrayObject* array, int print, npy_intp* shape, int ndim) {
    // Convert the weight into float    
    float* data = static_cast<float*>(PyArray_DATA(array));

    // Printing weights for checking it after conversion 
    if (print == 1){
        if (ndim == 2){
            for (npy_intp i = 0; i < shape[0]; ++i) {
                for (npy_intp j = 0; j < shape[1]; ++j) {
                    if (j < 5 || j > shape[1]-5){
                        printf("%" NPY_INTP_FMT" index %.5f ", j+(i*shape[1]), data[j+(i*shape[1])]);
                    }
                }
                printf("\n");
            }
        }
        else if (ndim == 1){
            for (npy_intp i = 0; i < shape[0]; ++i) {
                if (i < 5 || i > shape[0]-5){
                    printf("%" NPY_INTP_FMT" index %.5f ", i, data[i]);
                }
            }
            printf("\n");
        }
       
    }

    return data; 
}

double* move_weight_to_cuda(float* weights ,long int total_size){
    // Allocate CUDA device memory
    double* d_data;
    cudaMalloc((void**)&d_data, total_size * sizeof(double));

    // Copy the array data from host (CPU) to device (CUDA)
    cudaMemcpy(d_data, weights, total_size * sizeof(double), cudaMemcpyHostToDevice);
    return d_data;
}

double* read_weights(const char* file_path){
    PyArrayObject* array = read_weights_from_numpy(file_path, 1);
    if (array == nullptr) {
        // Handle error
        return nullptr;
    }

    int ndims = get_numpy_ndims(array);
    npy_intp* shape = get_numpy_shape(array);
    
    long int size = get_numpy_size(array);
    float* weights = convert_PyArrayObject_to_float(array, 1, shape, ndims);
    
    // Release the PyObject reference
    Py_DECREF(array);

    double* cuda_weights = move_weight_to_cuda(weights, size);
    return cuda_weights;
}


int main() {
    // Initialize the Python interpreter
    Py_Initialize();

    // Ensure that NumPy is available
    import_array();
    double* fc1_w = read_weights("./mlp/mnist_mlp/model/fc1.weight.npy");
    double* fc2_w = read_weights("./mlp/mnist_mlp/model/fc2.weight.npy");
    double* fc3_w = read_weights("./mlp/mnist_mlp/model/fc3.weight.npy");

    double* fc1_b = read_weights("./mlp/mnist_mlp/model/fc1.bias.npy");
    double* fc2_b = read_weights("./mlp/mnist_mlp/model/fc2.bias.npy");
    double* fc3_b = read_weights("./mlp/mnist_mlp/model/fc3.bias.npy");

    // Finalize the Python interpreter
    Py_Finalize();

    // Clean up CUDA device memory
    cudaFree(fc1_w);
    cudaFree(fc2_w);
    cudaFree(fc3_w);
    cudaFree(fc1_b);
    cudaFree(fc2_b);
    cudaFree(fc3_b);

    return 0;
}