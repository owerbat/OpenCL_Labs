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

    // OpenMP Block
    auto ompBlockTime = omp_gemm_block(n, a, b, c);
    print_matrix(c, n, m, "OpenMP Block result:");
    clear_matrix(c, n);

    // OpenCL GPU
    auto openCLGPUTime = opencl_gemm_gpu(n, a, b, c);
    print_matrix(c, n, m, "OpenCL GPU result:");
    clear_matrix(c, n);

    // OpenCL CPU
    auto openCLCPUTime = opencl_gemm_cpu(n, a, b, c);
    print_matrix(c, n, m, "OpenCL CPU result:");
    clear_matrix(c, n);

    // OpenCL GPU Block
    auto openCLGPUBlockTime = opencl_gemm_block_gpu(n, a, b, c);
    print_matrix(c, n, m, "OpenCL GPU Block result:");
    clear_matrix(c, n);

    // OpenCL GPU Block
    auto openCLCPUBlockTime = opencl_gemm_block_cpu(n, a, b, c);
    print_matrix(c, n, m, "OpenCL CPU Block result:");
    clear_matrix(c, n);

    // OpenCL GPU (image)
    auto openCLGPUImageTime = opencl_gemm_gpu_image(n, a, b, c);
    print_matrix(c, n, m, "OpenCL GPU (image) result:");
    clear_matrix(c, n);

    // OpenCL CPU (image)
    auto openCLCPUImageTime = opencl_gemm_cpu_image(n, a, b, c);
    print_matrix(c, n, m, "OpenCL CPU (image) result:");
    clear_matrix(c, n);

    // OpenCL GPU Block (image)
    auto openCLGPUBlockImageTime = opencl_gemm_block_gpu_image(n, a, b, c);
    print_matrix(c, n, m, "OpenCL GPU Block (image) result:");
    clear_matrix(c, n);

    // OpenCL GPU Block (image)
    auto openCLCPUBlockImageTime = opencl_gemm_block_cpu_image(n, a, b, c);
    print_matrix(c, n, m, "OpenCL CPU Block (image) result:");
    clear_matrix(c, n);

    // Total OpenMP
    std::cout << "\nTime OpenMP:\n"
              << "OpenMP       " << std::chrono::duration_cast<std::chrono::milliseconds>(ompTime).count() << " ms\n"
              << "OpenMP Block " << std::chrono::duration_cast<std::chrono::milliseconds>(ompBlockTime).count() << " ms\n";

    // Total OpenCL
    std::cout << "\nTime OpenCL (buffer):\n"
              << "OpenCL GPU       " << std::chrono::duration_cast<std::chrono::milliseconds>(openCLGPUTime).count() << " ms\n"
              << "OpenCL CPU       " << std::chrono::duration_cast<std::chrono::milliseconds>(openCLCPUTime).count() << " ms\n"
              << "OpenCL GPU Block " << std::chrono::duration_cast<std::chrono::milliseconds>(openCLGPUBlockTime).count() << " ms\n"
              << "OpenCL CPU Block " << std::chrono::duration_cast<std::chrono::milliseconds>(openCLCPUBlockTime).count() << " ms\n";

    // Total OpenCL with images instead of buffers
    std::cout << "\nTime OpenCL (image):\n"
              << "OpenCL GPU       " << std::chrono::duration_cast<std::chrono::milliseconds>(openCLGPUImageTime).count() << " ms\n"
              << "OpenCL CPU       " << std::chrono::duration_cast<std::chrono::milliseconds>(openCLCPUImageTime).count() << " ms\n"
              << "OpenCL GPU Block " << std::chrono::duration_cast<std::chrono::milliseconds>(openCLGPUBlockImageTime).count() << " ms\n"
              << "OpenCL CPU Block " << std::chrono::duration_cast<std::chrono::milliseconds>(openCLCPUBlockImageTime).count() << " ms\n";

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
