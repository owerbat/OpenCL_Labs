#include "jacobi.h"

bool checkMatrix(size_t size, float *a);
bool checkSolution(size_t size, float *a, float *b, float *x1, float *check);

int main() {
    const size_t size = 1 << 8;
    std::cout << "size = " << size << std::endl;

    float *a     = new float[size * size];
    float *b     = new float[size];
    float *x0    = new float[size];
    float *x1    = new float[size];
    float *norm  = new float[size];
    float *check = new float[size];

    for (size_t i = 0; i < size; ++i)
        for (size_t j = 0; j < size; ++j)
            a[i * size + j] = (j == i) ? 100 : (rand() % 5 + 1) / (1.f * size);

    if (!checkMatrix(size, a)) {
        std::cout << "Error: incorrect matrix" << std::endl;
        return 1;
    }

    for (size_t i = 0; i < size; ++i)
        b[i] = (rand() % 5 + 1) / (1.f * size);

    // OpenCL GPU
    auto openCLGPUTime = opencl_jacobi_gpu(size, a, b, x0, x1, norm);
    std::cout << checkSolution(size, a, b, x1, check) ? "GPU: PASSED\n" : "GPU: FAILED\n";

    // OpenCL CPU
    auto openCLCPUTime = opencl_jacobi_cpu(size, a, b, x0, x1, norm);
    std::cout << checkSolution(size, a, b, x1, check) ? "CPU: PASSED\n" : "CPU: FAILED\n";

    // Total OpenCL
    std::cout << "\nTime OpenCL (buffer):\n"
              << "OpenCL GPU " << std::chrono::duration_cast<std::chrono::milliseconds>(openCLGPUTime).count() << " ms\n"
              << "OpenCL CPU " << std::chrono::duration_cast<std::chrono::milliseconds>(openCLCPUTime).count() << " ms\n";

    return 0;
}

bool checkMatrix(size_t size, float *a) {
    float sum;
    for (size_t i = 0; i < size; ++i) {
        sum = 0;
        for (size_t j = 0; j < i; ++j)
            sum += a[i * size + j];
        for (size_t j = i + 1; j < size; ++j)
            sum += a[i * size + j];
        if (sum > a[i * size + i])
            return false;
    }
    return true;
}

bool checkSolution(size_t size, float *a, float *b, float *x1, float *check) {
    for (size_t i = 0; i < size; i++) {
        check[i] = 0;
        for (size_t j = 0; j < size; j++) {
            check[i] += a[i * size + j] * x1[j];
        }
    }

    float sum = 0.0f;
    for (size_t k = 0; k < size; k++) {
        sum += (check[k] - b[k]) * (check[k] - b[k]);
    }

    return sqrt(sum) < 1e-4f;
}
