#include "gemm.h"


void print_matrix(const float *matrix, const cl_uint size, const cl_uint m, const char *message);
void clear_matrix(float *matrix, const cl_uint size);


int main() {
    const cl_uint n = BLOCK_SIZE * (2 << 4), m = 5;
    cl_int i, j;
    float *a = new float[n * n], *b = new float[n * n], *c = new float[n * n];

    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            a[i * n + j] = (i == j) ? 1.0f : 0.0f;
            b[i * n + j] = (i == j) ? 2.0f : 0.0f;
            c[i * n + j] = 0.0f;
        }
    }

    // OpenMP
    auto ompTime = omp_gemm(n, a, b, c);
    print_matrix(c, n, m, "OpenMP result:");
    clear_matrix(c, n);

    // OpenCL GPU
    auto openCLGPUTime = opencl_gemm_gpu(n, a, b, c);
    print_matrix(c, n, m, "OpenCL GPU result:");
    clear_matrix(c, n);

    // OpenCL CPU
    auto openCLCPUTime = opencl_gemm_cpu(n, a, b, c);
    print_matrix(c, n, m, "OpenCL CPU result:");
    clear_matrix(c, n);

    // Total
    std::cout << "Time:\n"
              << "OpenMP     " << std::chrono::duration_cast<std::chrono::milliseconds>(ompTime).count() << " ms\n"
              << "OpenCL GPU " << std::chrono::duration_cast<std::chrono::milliseconds>(openCLGPUTime).count() << " ms\n"
              << "OpenCL CPU " << std::chrono::duration_cast<std::chrono::milliseconds>(openCLCPUTime).count() << " ms\n";

    delete[] a, b, c;

    return 0;
}


void print_matrix(const float *matrix, const cl_uint size, const cl_uint bound, const char *message) {
    std::cout << message << std::endl;
    for (cl_uint i = 0; i < bound; ++i) {
        for (cl_uint j = 0; j < bound; ++j)
            std::cout << " " << matrix[i * size + j];
        std::cout << std::endl;
    }
}


void clear_matrix(float *matrix, const cl_uint size) {
    for (cl_uint i = 0; i < size; ++i)
        for (cl_uint j = 0; j < size; ++j)
            matrix[i * size + j] = 0.0f;
}
